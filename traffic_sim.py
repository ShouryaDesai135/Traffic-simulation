# File: traffic_sim.py - COMPLETE OPTIMIZED VERSION
import random
import time
import threading
import pygame
import sys
import os
import pandas as pd
import pickle
from collections import deque

# ========== METRICS & EVENTS ==========
metrics = {
    "simulation_time": 0,
    "vehicles_passed": 0,
    "throughput": 0.0,
    "emergency_active": False,
    "emergency_lane": None,
    "ml_active": False,
    "signals": [],
    "lanes": {},
    "priority_queue": {},
    "events": []
}

event_log = deque(maxlen=200)

def add_event(message, etype="info"):
    global timeElapsed, event_log, metrics
    try:
        event_log.append({
            "time": int(timeElapsed) if "timeElapsed" in globals() else 0,
            "type": etype,
            "message": message
        })
        metrics["events"] = list(event_log)[-50:]
    except Exception as e:
        print(f"[EventLogError] {e}")

def update_metrics():
    global timeElapsed, signals, vehicles, emergencyMode, emergencyLane, currentGreen, currentYellow, ML_ENABLED, priority_queue, directionNumbers
    
    try:
        metrics["simulation_time"] = int(timeElapsed)
        metrics["vehicles_passed"] = sum(v.get("crossed", 0) for v in vehicles.values())
        metrics["throughput"] = round(metrics["vehicles_passed"] / max(1, timeElapsed), 2)
        metrics["emergency_active"] = emergencyMode
        metrics["emergency_lane"] = emergencyLane
        metrics["ml_active"] = ML_ENABLED
        
        signal_states = []
        for i in range(len(signals)):
            state = "red"
            timer = signals[i].red
            
            if i == currentGreen:
                if currentYellow == 1:
                    state = "yellow"
                    timer = signals[i].yellow
                else:
                    state = "green"
                    timer = signals[i].green
            
            signal_states.append({
                "state": state,
                "timer": max(0, int(timer)),
                "total_green_time": int(signals[i].totalGreenTime)
            })
        
        metrics["signals"] = signal_states
        
        lane_info = {}
        for direction in ['right', 'down', 'left', 'up']:
            waiting = sum(len(vehicles[direction][i]) for i in range(3))
            crossed = vehicles[direction].get('crossed', 0)
            
            if waiting >= 15:
                congestion = "High"
            elif waiting >= 6:
                congestion = "Medium"
            else:
                congestion = "Low"
            
            lane_info[direction] = {
                "waiting": waiting,
                "crossed": crossed,
                "total": waiting + crossed,
                "congestion": congestion
            }
        
        metrics["lanes"] = lane_info
        
        pq_data = {}
        for i in range(4):
            direction = directionNumbers[i]
            pq = priority_queue.lane_data[i]
            pq_data[direction] = {
                "priority_score": int(pq['priority_score']),
                "wait_time": int(pq['waiting_time']),
                "efficiency": round(1.0, 2),
                "clearance_rate": round(lane_info[direction]['waiting'] / max(1, timeElapsed) * 10, 2),
                "predicted_clear_time": int(lane_info[direction]['waiting'] * 2)
            }
        
        metrics["priority_queue"] = pq_data
        
    except Exception as e:
        print(f"[MetricsError] {e}")

# ============= ML MODEL LOADING =============
try:
    import joblib
    try:
        ml_model = joblib.load('model/traffic_model_4way.pkl')
        anomaly_detector = joblib.load('model/anomaly_detector_4way.pkl')
    except:
        with open('model/traffic_model_4way.pkl', 'rb') as f:
            ml_model = pickle.load(f, encoding='latin1')
        with open('model/anomaly_detector_4way.pkl', 'rb') as f:
            anomaly_detector = pickle.load(f, encoding='latin1')
    
    print("✅ ML Model loaded successfully!")
    ML_ENABLED = True
except Exception as e:
    print(f"⚠️ ML Model loading failed: {e}")
    ML_ENABLED = False
    ml_model = None
    anomaly_detector = None

# ============= DATASET LOADING =============
try:
    dataset = pd.read_csv('data/predictions_4way_full.csv')
    dataset_index = 0
    DATASET_ENABLED = True
    print(f"✅ Dataset loaded: {len(dataset)} snapshots")
except Exception as e:
    try:
        dataset = pd.read_csv('data/predictions_4way_by_lane.csv')
        dataset_index = 0
        DATASET_ENABLED = True
        print(f"✅ Dataset loaded: {len(dataset)} snapshots")
    except:
        try:
            dataset = pd.read_csv('data/traffic_dataset_4way.csv')
            dataset_index = 0
            DATASET_ENABLED = True
            print(f"✅ Original dataset loaded: {len(dataset)} snapshots")
        except:
            print(f"⚠️ No dataset found")
            DATASET_ENABLED = False

# Default values
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300
timeElapsed = 0

currentGreen = 0
nextGreen = 1
currentYellow = 0

emergencyMode = False
emergencyLane = None
emergencyVehicleRef = None

speeds = {
    'car': 2.25, 
    'bus': 1.8, 
    'truck': 1.8, 
    'rickshaw': 2, 
    'bike': 2.5,
    'ambulance': 3.5
}

ambulance_spawned = {0: False, 1: False, 2: False, 3: False}

# ============= PRIORITY QUEUE =============
class LanePriorityQueue:
    def __init__(self):
        self.lane_data = {i: {
            'waiting_vehicles': 0,
            'waiting_time': 0,
            'emergency_vehicles': 0,
            'last_served': time.time(),
            'congestion_level': 'Low',
            'priority_score': 0,
            'starvation_factor': 0
        } for i in range(4)}
        self.max_wait_time = 120
        self.fairness_threshold = 60

    def update_lane(self, lane_idx, vehicle_count, emergency_count, congestion):
        data = self.lane_data[lane_idx]
        data['waiting_vehicles'] = vehicle_count
        data['emergency_vehicles'] = emergency_count
        data['congestion_level'] = congestion

        current_time = time.time()
        time_since_served = current_time - data['last_served']
        data['waiting_time'] = time_since_served

        if time_since_served > self.fairness_threshold:
            data['starvation_factor'] = (time_since_served - self.fairness_threshold) / 10
        else:
            data['starvation_factor'] = 0

        self.calculate_priority(lane_idx)

    def calculate_priority(self, lane_idx):
        data = self.lane_data[lane_idx]
        score = 0

        if data['emergency_vehicles'] > 0:
            score += 10000

        congestion_weights = {'High': 500, 'Medium': 200, 'Low': 50}
        score += congestion_weights.get(data['congestion_level'], 200)
        score += data['waiting_vehicles'] * 15
        score += data['waiting_time'] * 2
        score += data['starvation_factor'] * 50

        if data['waiting_time'] > self.max_wait_time:
            score += 5000

        data['priority_score'] = score

    def get_next_green(self, current_green):
        for lane_idx in range(4):
            if self.lane_data[lane_idx]['emergency_vehicles'] > 0:
                if lane_idx != current_green:
                    return lane_idx

        candidates = []
        for lane_idx in range(4):
            if lane_idx != current_green:
                score = self.lane_data[lane_idx]['priority_score']
                waiting_time = self.lane_data[lane_idx]['waiting_time']
                candidates.append((lane_idx, score, waiting_time))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            return candidates[0][0]

        return (current_green + 1) % noOfSignals

    def mark_served(self, lane_idx):
        self.lane_data[lane_idx]['last_served'] = time.time()
        self.lane_data[lane_idx]['waiting_time'] = 0
        self.lane_data[lane_idx]['starvation_factor'] = 0

priority_queue = LanePriorityQueue()

# Coordinates
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {
    'right': {0:[], 1:[], 2:[], 'crossed':0}, 
    'down': {0:[], 1:[], 2:[], 'crossed':0}, 
    'left': {0:[], 1:[], 2:[], 'crossed':0}, 
    'up': {0:[], 1:[], 2:[], 'crossed':0}
}

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike', 5: 'ambulance'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}
originMapping = {
    'North': 'up', 'East': 'right', 'South': 'down', 'West': 'left',
    'NORTH': 'up', 'EAST': 'right', 'SOUTH': 'down', 'WEST': 'left',
    'north': 'up', 'east': 'right', 'south': 'down', 'west': 'left'
}
simulationToDataset = {'right': 'East', 'down': 'South', 'left': 'West', 'up': 'North'}
directionToNumber = {'right': 0, 'down': 1, 'left': 2, 'up': 3}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}
mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}

rotationAngle = 3
gap = 15
gap2 = 15

pygame.init()
simulation = pygame.sprite.Group()

spawn_counter = 0
SPAWN_PER_AMBULANCE = 60
detectionTime = 5

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = int(red)
        self.yellow = int(yellow)
        self.green = int(green)
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.signalText = "30"
        self.totalGreenTime = 0
        self.mlPriority = 0

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.isEmergency = (vehicleClass == 'ambulance')
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        path = "images/" + direction + "/" + vehicleClass + ".png"

        if not os.path.exists(path) and vehicleClass == 'ambulance':
            path = "images/" + direction + "/car.png"
            self.originalImage = pygame.image.load(path)
            self.originalImage = self.originalImage.copy()
            self.originalImage.fill((255, 100, 100), special_flags=pygame.BLEND_RGB_MULT)
        else:
            self.originalImage = pygame.image.load(path)

        self.currentImage = self.originalImage.copy()

        try:
            car_path = "images/" + direction + "/car.png"
            if os.path.exists(car_path):
                ref_img = pygame.image.load(car_path)
                ref_rect = ref_img.get_rect()
                if self.vehicleClass == 'ambulance':
                    self.originalImage = pygame.transform.scale(self.originalImage, (ref_rect.width, ref_rect.height))
        except:
            pass

        self.currentImage = self.originalImage.copy()

        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        
        simulation.add(self)

        if self.isEmergency:
            ambulance_spawned[self.direction_number] = True
            activateEmergencyMode(self.direction_number, self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.isEmergency:
                    print(f"✅ Ambulance crossed in {self.direction.upper()}")
                    add_event(f"Ambulance cleared: {self.direction.upper()}", etype="info")
            
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if ((self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and 
                        (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or 
                         vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                        self.x += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or 
                            self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if ((self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and 
                    (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or 
                     vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.isEmergency:
                    print(f"✅ Ambulance crossed in {self.direction.upper()}")
                    add_event(f"Ambulance cleared: {self.direction.upper()}", etype="info")
            
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if ((self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and 
                        (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or 
                         vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                        self.y += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or 
                            self.y < (vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if ((self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and 
                    (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or 
                     vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.isEmergency:
                    print(f"✅ Ambulance crossed in {self.direction.upper()}")
                    add_event(f"Ambulance cleared: {self.direction.upper()}", etype="info")
            
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if ((self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and 
                        (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or 
                         vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                        self.x -= self.speed
                else: 
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or 
                            self.x > (vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if ((self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and 
                    (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or 
                     vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.isEmergency:
                    print(f"✅ Ambulance crossed in {self.direction.upper()}")
                    add_event(f"Ambulance cleared: {self.direction.upper()}", etype="info")
            
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if ((self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and 
                        (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or 
                         vehicles[self.direction][self.lane][self.index-1].turned == 1)):
                        self.y -= self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if (self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or 
                            self.y > (vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if ((self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and 
                    (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or 
                     vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.y -= self.speed

# ============= EMERGENCY HANDLING =============
def checkEmergency():
    global emergencyMode, emergencyLane, emergencyVehicleRef

    if emergencyMode and emergencyVehicleRef:
        if emergencyVehicleRef.crossed == 1:
            deactivateEmergencyMode()
            return

    for dir_idx, direction in directionNumbers.items():
        for lane in [0, 1, 2]:
            for vehicle in list(vehicles[direction][lane]):
                if vehicle.isEmergency and vehicle.crossed == 0:
                    if not emergencyMode:
                        print(f"🚨 EMERGENCY! Ambulance in {direction.upper()}")
                        add_event(f"🚨 EMERGENCY vehicle: {direction.upper()}", etype="emergency")
                        emergencyMode = True
                        emergencyLane = direction
                        emergencyVehicleRef = vehicle
                        ambulance_spawned[dir_idx] = True
                        activateEmergencyMode(dir_idx, vehicle)
                        return

def activateEmergencyMode(emergency_direction, ambulance_vehicle=None):
    global currentGreen, currentYellow, signals, emergencyMode, emergencyLane, emergencyVehicleRef

    print(f"🚨 Emergency mode: {directionNumbers[emergency_direction].upper()}")
    add_event(f"🚨 Emergency PREEMPTION: {directionNumbers[emergency_direction].upper()}", etype='emergency')

    emergencyMode = True
    emergencyLane = directionNumbers.get(emergency_direction, None)
    emergencyVehicleRef = ambulance_vehicle

    for i in range(noOfSignals):
        if i != emergency_direction:
            signals[i].green = 0
            signals[i].yellow = 0
            signals[i].red = 999

    signals[emergency_direction].green = max(20, int(defaultGreen))
    signals[emergency_direction].red = 0
    signals[emergency_direction].yellow = 0
    currentGreen = emergency_direction
    currentYellow = 0

    try:
        metrics['emergency_active'] = True
        metrics['emergency_lane'] = directionNumbers[emergency_direction]
        update_metrics()
    except:
        pass

def deactivateEmergencyMode():
    global emergencyMode, emergencyLane, emergencyVehicleRef, currentGreen, nextGreen, signals

    print("✅ Emergency cleared. Resuming control.")
    add_event("✅ Emergency cleared!", etype='info')
    
    emergencyMode = False
    old_emergency_lane = emergencyLane
    emergencyLane = None
    emergencyVehicleRef = None

    for k in ambulance_spawned.keys():
        ambulance_spawned[k] = False

    # CRITICAL: Force signal to expire and switch
    signals[currentGreen].green = 2
    
    for i in range(noOfSignals):
        if i != currentGreen:
            signals[i].red = defaultRed
        signals[i].yellow = defaultYellow

    try:
        if old_emergency_lane:
            dir_num = directionToNumber[old_emergency_lane]
            priority_queue.mark_served(dir_num)
    except:
        pass

    try:
        updatePriorityQueue()
        nextGreen = priority_queue.get_next_green(currentGreen)
        setTimeML()
    except Exception as e:
        nextGreen = (currentGreen + 1) % noOfSignals

    try:
        metrics['emergency_active'] = False
        metrics['emergency_lane'] = None
        update_metrics()
    except:
        pass

    print(f"🔄 Switching to {directionNumbers[nextGreen].upper()} in 2s")

# ============= ML FUNCTIONS =============
def updatePriorityQueue():
    global priority_queue

    for direction_num in range(noOfSignals):
        direction = directionNumbers[direction_num]
        vehicle_count = sum(len(vehicles[direction][i]) for i in range(3))
        emergency_count = sum(1 for i in range(3) for v in vehicles[direction][i] 
                            if hasattr(v, 'isEmergency') and v.isEmergency and v.crossed == 0)
        prediction = getMLPrediction(direction_num)
        congestion = prediction['congestion'] if prediction else 'Low'
        priority_queue.update_lane(direction_num, vehicle_count, emergency_count, congestion)

def getMLPrediction(direction_num):
    global dataset, dataset_index, DATASET_ENABLED

    if not DATASET_ENABLED:
        direction = directionNumbers[direction_num]
        vehicle_count = sum(len(vehicles[direction][i]) for i in range(3))

        if vehicle_count >= 15:
            congestion = 'High'
        elif vehicle_count >= 6:
            congestion = 'Medium'
        else:
            congestion = 'Low'

        return {
            'congestion': congestion,
            'vehicle_count': vehicle_count,
            'emergency_count': 0,
            'priority': vehicle_count * 15,
            'features': {'vehicle_count': vehicle_count}
        }

    try:
        direction_name = directionNumbers[direction_num]
        dataset_origin = simulationToDataset[direction_name]
        lane_data = dataset[dataset['origin'] == dataset_origin]

        if len(lane_data) == 0:
            return None

        snapshot = lane_data.iloc[dataset_index % len(lane_data)]

        features = {
            'vehicle_count': int(snapshot.get('vehicle_count', 0)),
            'avg_speed': float(snapshot.get('avg_speed', 0)),
            'emergency_count': int(snapshot.get('emergency_count', 0)),
            'left_turn_queue': int(snapshot.get('left_turn_queue', 0)),
            'straight_queue': int(snapshot.get('straight_queue', 0)),
            'right_turn_queue': int(snapshot.get('right_turn_queue', 0)),
            'ml_anomaly_score': float(snapshot.get('ml_anomaly_score', 0))
        }

        congestion = snapshot.get('predicted_congestion', 'Medium')
        if pd.isna(congestion):
            congestion = snapshot.get('congestion_level', 'Medium')

        priority = float(snapshot.get('priority_score', 0))
        if pd.isna(priority) or priority == 0:
            priority = calculateMLPriority(congestion, features)

        return {
            'congestion': congestion,
            'vehicle_count': features['vehicle_count'],
            'emergency_count': features['emergency_count'],
            'priority': priority,
            'features': features
        }

    except Exception as e:
        return None

def calculateMLPriority(congestion, features):
    priority = 0
    congestion_weights = {'High': 500, 'Medium': 200, 'Low': 50}
    priority += congestion_weights.get(congestion, 200)
    priority += features.get('vehicle_count', 0) * 15
    priority += features.get('waiting_time', 0) * 2
    return priority

def setTimeML():
    global nextGreen, signals, emergencyMode, currentGreen

    if emergencyMode:
        return

    updatePriorityQueue()
    nextGreen = priority_queue.get_next_green(currentGreen)
    prediction = getMLPrediction(nextGreen)

    if prediction:
        congestion = prediction['congestion']
        vehicle_count = prediction['vehicle_count']
        emergency_count = prediction['emergency_count']

        if congestion == 'High':
            greenTime = defaultMaximum
        elif congestion == 'Medium':
            greenTime = (defaultMinimum + defaultMaximum) // 2
        else:
            greenTime = defaultMinimum

        if congestion == 'High':
            greenTime = min(defaultMaximum, greenTime + 5)
        elif congestion == 'Low' and vehicle_count < 3:
            greenTime = max(defaultMinimum, greenTime - 5)

        if emergency_count > 0:
            greenTime = min(greenTime + 10, defaultMaximum)

        signals[nextGreen].mlPriority = vehicle_count
        direction_name = directionNumbers[nextGreen]
        print(f"🧠 ML: {direction_name.upper()} v={vehicle_count} c={congestion} t={greenTime}s")
    else:
        direction = directionNumbers[nextGreen]
        vehicle_count = sum(len(vehicles[direction][i]) for i in range(3))
        greenTime = max(defaultMinimum, min(defaultMaximum, vehicle_count * 3))

    signals[nextGreen].green = int(greenTime)
    prev = (nextGreen-1) % noOfSignals
    signals[nextGreen].red = max(0, int(signals[prev].yellow) + int(signals[prev].green))

def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

def repeat():
    global currentGreen, currentYellow, nextGreen, emergencyMode

    while signals[currentGreen].green > 0:
        printStatus()
        updateValues()
        checkEmergency()

        if not emergencyMode:
            dir_name = directionNumbers[currentGreen]
            uncrossed = sum(1 for lane in [0,1,2] for v in vehicles[dir_name][lane] if v.crossed == 0)

            if uncrossed == 0 and signals[currentGreen].green > 3:
                print(f"⏭️ {dir_name.upper()} cleared early")
                signals[currentGreen].green = 3

        if not emergencyMode:
            next_idx = (currentGreen+1)%(noOfSignals)
            if signals[next_idx].red == detectionTime:
                thread = threading.Thread(name="detection", target=setTimeML, args=())
                thread.daemon = True
                thread.start()

        time.sleep(1)

    currentYellow = 1

    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]

    while signals[currentGreen].yellow > 0:
        printStatus()
        updateValues()
        checkEmergency()
        time.sleep(1)

    currentYellow = 0

    if not emergencyMode:
        priority_queue.mark_served(currentGreen)

    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen
    nextGreen = (currentGreen+1)%noOfSignals
    signals[nextGreen].red = max(0, signals[currentGreen].yellow + signals[currentGreen].green)
    repeat()

def printStatus():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                status = " GREEN"
            else:
                status = "YELLOW"
        else:
            status = "   RED"

        pq_data = priority_queue.lane_data[i]
        wait_time = int(pq_data['waiting_time'])
        priority_score = int(pq_data['priority_score'])

        ml_info = f" [P:{priority_score} W:{wait_time}s]"
        r = max(0, int(signals[i].red))
        y = max(0, int(signals[i].yellow))
        g = max(0, int(signals[i].green))
        print(f"{status} TS{i+1} ({directionNumbers[i]:>5})-> r:{r:>3} y:{y} g:{g:>2}{ml_info}")

    if emergencyMode:
        print(f"🚨 EMERGENCY: {emergencyLane.upper()}")
    print()

def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                if signals[i].green > 0:
                    signals[i].green -= 1
                    signals[i].totalGreenTime += 1
            else:
                if signals[i].yellow > 0:
                    signals[i].yellow -= 1
        else:
            if signals[i].red > 0:
                signals[i].red -= 1

# ============= VEHICLE GENERATION =============
def generateVehiclesFromDataset():
    global dataset, dataset_index, DATASET_ENABLED, spawn_counter, SPAWN_PER_AMBULANCE

    if not DATASET_ENABLED:
        generateVehiclesRandom()
        return

    print("🚗 Vehicle generation started")

    while True:
        try:
            spawns_this_cycle = random.randint(2, 6)

            for _ in range(spawns_this_cycle):
                spawn_counter += 1
                
                if spawn_counter % SPAWN_PER_AMBULANCE == 0:
                    direction = random.choice(list(directionNumbers.values()))
                    lane_number = random.randint(1, 2)
                    
                    print(f"🚨 Ambulance: {direction.upper()} L{lane_number}")
                    add_event(f"🚨 Ambulance: {direction.upper()}", etype="emergency")
                    
                    Vehicle(lane_number, 'ambulance', directionToNumber[direction], direction, 0)
                    time.sleep(0.5)
                    continue

                snapshot_row = dataset.iloc[dataset_index % len(dataset)]
                
                if random.random() < 0.6:
                    snapshot = snapshot_row
                else:
                    snapshot = dataset.sample(n=1).iloc[0]

                origin = snapshot.get('origin', 'East')
                sim_dir = originMapping.get(origin, 'right')

                if random.random() < 0.7:
                    direction = sim_dir
                else:
                    direction = random.choice(list(directionNumbers.values()))

                vehicle_type = random.choice(['car','car','bus','truck','bike'])
                if vehicle_type == 'bike':
                    lane_number = 0
                else:
                    lane_number = random.randint(1, 2)

                will_turn = 0
                if lane_number == 2 and random.random() < 0.3:
                    will_turn = 1

                Vehicle(lane_number, vehicle_type, directionToNumber[direction], direction, will_turn)
                time.sleep(random.uniform(0.08, 0.18))

                if random.random() < 0.5:
                    dataset_index += 1

            time.sleep(random.uniform(2.5, 3.6))

        except Exception as e:
            print(f"❌ Generation error: {e}")
            time.sleep(1)

def generateVehiclesRandom():
    print("🚗 Random generation started")
    global spawn_counter, SPAWN_PER_AMBULANCE

    while True:
        spawn_counter += 1
        if spawn_counter % SPAWN_PER_AMBULANCE == 0:
            vehicle_type = 5
        else:
            vehicle_type = random.randint(0, 4)

        if vehicle_type == 5:
            lane_number = random.randint(1, 2)
            will_turn = 0
        elif vehicle_type == 4:
            lane_number = 0
            will_turn = 0
        else:
            lane_number = random.randint(1, 2)
            will_turn = 0
            if lane_number == 2:
                will_turn = 1 if random.randint(0, 4) <= 2 else 0

        temp = random.randint(0, 999)
        a = [400, 800, 900, 1000]
        if temp < a[0]:
            direction_number = 0
        elif temp < a[1]:
            direction_number = 1
        elif temp < a[2]:
            direction_number = 2
        else:
            direction_number = 3

        try:
            Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, 
                    directionNumbers[direction_number], will_turn)
        except Exception as e:
            print(f"Spawn error: {e}")

        time.sleep(random.uniform(0.5, 0.9) if vehicle_type != 5 else random.uniform(0.35, 0.55))

def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = sum(vehicles[directionNumbers[i]]['crossed'] for i in range(noOfSignals))
            
            print('\n' + '='*50)
            print('SIMULATION COMPLETE')
            print('='*50)
            print('Lane-wise Vehicle Counts:')
            for i in range(noOfSignals):
                crossed = vehicles[directionNumbers[i]]['crossed']
                print(f'  {directionNumbers[i].upper():>5} (Lane {i+1}): {crossed} vehicles')

            print(f'\nTotal: {totalVehicles} vehicles')
            print(f'Time: {timeElapsed}s')
            print(f'Throughput: {float(totalVehicles)/float(timeElapsed):.2f} v/s')
            print(f'ML: {"ACTIVE" if ML_ENABLED else "INACTIVE"}')
            print('='*50)
            os._exit(1)

# ============= MAIN CLASS (DON'T AUTO-RUN) =============
class Main:
    @staticmethod
    def run():
        """Main simulation loop - only runs when explicitly called"""
        
        thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
        thread4.daemon = True
        thread4.start()

        thread2 = threading.Thread(name="initialization", target=initialize, args=())
        thread2.daemon = True
        thread2.start()

        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        yellow = (255, 255, 0)

        screenWidth = 1400
        screenHeight = 800
        screenSize = (screenWidth, screenHeight)

        background = pygame.image.load('images/mod_int.png')
        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption("ML Traffic Signal Simulation")

        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
        font = pygame.font.Font(None, 30)
        emergencyFont = pygame.font.Font(None, 40)
        smallFont = pygame.font.Font(None, 24)

        if DATASET_ENABLED:
            thread3 = threading.Thread(name="generateVehicles", target=generateVehiclesFromDataset, args=())
        else:
            thread3 = threading.Thread(name="generateVehicles", target=generateVehiclesRandom, args=())

        thread3.daemon = True
        thread3.start()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            screen.blit(background, (0, 0))

            for i in range(0, noOfSignals):
                g = max(0, int(signals[i].green))
                y = max(0, int(signals[i].yellow))
                r = max(0, int(signals[i].red))

                if i == currentGreen:
                    if currentYellow == 1:
                        signals[i].signalText = y if y > 0 else "STOP"
                        screen.blit(yellowSignal, signalCoods[i])
                    else:
                        signals[i].signalText = g if g > 0 else "SLOW"
                        screen.blit(greenSignal, signalCoods[i])
                else:
                    signals[i].signalText = r if r <= 10 and r > 0 else "---"
                    screen.blit(redSignal, signalCoods[i])

            for i in range(0, noOfSignals):
                signalText = font.render(str(signals[i].signalText), True, white, black)
                screen.blit(signalText, signalTimerCoods[i])

            if emergencyMode:
                flash_alpha = 220 if int(timeElapsed * 2) % 2 == 0 else 180
                emergency_bg = pygame.Surface((350, 120))
                emergency_bg.fill(red)
                emergency_bg.set_alpha(flash_alpha)
                screen.blit(emergency_bg, (5, 5))
                
                emergencyText = emergencyFont.render("🚨 EMERGENCY 🚨", True, white)
                screen.blit(emergencyText, (15, 15))

                if emergencyLane:
                    laneText = emergencyFont.render(f"{emergencyLane.upper()} LANE", True, yellow)
                    screen.blit(laneText, (15, 60))
                    
                    statusText = smallFont.render("Ambulance clearing...", True, white)
                    screen.blit(statusText, (15, 95))

            ml_y_pos = 130 if emergencyMode else 20
            if ML_ENABLED:
                mlText = smallFont.render("🧠 ML ACTIVE", True, green, black)
                screen.blit(mlText, (20, ml_y_pos))

            stats_y = ml_y_pos + 30
            statsTitle = smallFont.render("Lane Priorities:", True, white, black)
            screen.blit(statsTitle, (20, stats_y))

            for i in range(noOfSignals):
                pq_data = priority_queue.lane_data[i]
                priority_score = int(pq_data['priority_score'])
                wait_time = int(pq_data['waiting_time'])
                vehicles_waiting = pq_data['waiting_vehicles']

                color = green if i == currentGreen else white
                statsText = smallFont.render(
                    f"{directionNumbers[i].upper()}: P={priority_score} W={wait_time}s V={vehicles_waiting}", 
                    True, color, black
                )
                screen.blit(statsText, (20, stats_y + 25 + i*25))

            for vehicle in list(simulation):
                try:
                    screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
                    vehicle.move()
                except:
                    continue

            pygame.display.update()


# ============= ENTRY POINT =============
if __name__ == "__main__":
    print("=" * 70)
    print("🚦 ML TRAFFIC CONTROL SIMULATION")
    print("=" * 70)
    print("Starting simulation...")
    Main.run()