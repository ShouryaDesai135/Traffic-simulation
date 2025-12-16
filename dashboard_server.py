import os
import sys
import threading
import time
import json
from flask import Flask, jsonify, send_from_directory

# Prevent traffic_sim from auto-starting when imported
os.environ['TRAFFIC_SIM_NO_AUTOSTART'] = '1'

# Import traffic_sim for metrics access only
try:
    import traffic_sim
    print("✅ Traffic simulation module loaded")
except Exception as e:
    print(f"⚠️ Warning: Could not import traffic_sim: {e}")
    traffic_sim = None

# Create Flask app with explicit template folder
app = Flask(__name__, 
            template_folder='templates',
            static_folder='.',
            static_url_path='')

# ------------------ ROUTES ------------------

@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    try:
        return send_from_directory('templates', 'dashboard.html')
    except Exception as e:
        return f"""
        <h1>Error loading dashboard</h1>
        <p>{e}</p>
        <p>Make sure dashboard.html is in the 'templates/' folder</p>
        """

@app.route('/metrics.json')
def metrics():
    """Serve live simulation metrics as JSON for dashboard updates."""
    try:
        if traffic_sim:
            traffic_sim.update_metrics()
            return jsonify(traffic_sim.metrics)
        else:
            return jsonify({
                "error": "Simulation not running",
                "simulation_time": 0,
                "vehicles_passed": 0,
                "throughput": 0.0,
                "signals": [],
                "lanes": {},
                "priority_queue": {},
                "events": []
            })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "simulation_loaded": traffic_sim is not None
    })

# ------------------ BACKGROUND UPDATER ------------------

def update_metrics_loop():
    """Continuously refresh metrics every second."""
    print("📊 Metrics updater started")
    while True:
        try:
            if traffic_sim:
                traffic_sim.update_metrics()
        except Exception as e:
            print(f"[MetricsLoopError] {e}")
        time.sleep(1)

# ------------------ SERVER RUNNER ------------------

def run_server():
    """Function to start Flask server"""
    # Start background metrics updater
    updater = threading.Thread(target=update_metrics_loop, daemon=True)
    updater.start()
    
    print("\n" + "=" * 70)
    print("✅ FLASK DASHBOARD SERVER READY")
    print("=" * 70)
    print("🌐 Dashboard URL: http://127.0.0.1:5000")
    print("📊 Metrics endpoint: http://127.0.0.1:5000/metrics.json")
    print("💚 Health check: http://127.0.0.1:5000/health")
    print("\n⚠️  Keep this terminal open while using the dashboard")
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    # Run Flask app
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# ------------------ MAIN ENTRY ------------------

if __name__ == '__main__':
    print("🚀 Starting dashboard server...")
    run_server()