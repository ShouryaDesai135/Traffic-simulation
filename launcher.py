#!/usr/bin/env python3
"""
Unified launcher - Runs both simulation and dashboard in ONE process
No separate terminals needed!
"""

import threading
import time
import webbrowser
import sys

def start_flask_dashboard():
    """Start Flask server in background thread"""
    print("🌐 Starting Flask dashboard server...")
    
    from flask import Flask, jsonify, send_from_directory
    import traffic_sim
    
    app = Flask(__name__, template_folder='templates', static_folder='.', static_url_path='')
    
    @app.route('/')
    def index():
        try:
            return send_from_directory('templates', 'dashboard.html')
        except Exception as e:
            return f"Error: {e}. Make sure dashboard.html is in templates/ folder"
    
    @app.route('/metrics.json')
    def metrics():
        try:
            traffic_sim.update_metrics()
            return jsonify(traffic_sim.metrics)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "simulation_running": True})
    
    print("✅ Flask server ready at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

def start_simulation():
    """Start pygame simulation in main thread"""
    print("🚦 Starting traffic simulation...")
    import traffic_sim
    
    # Wait a moment for Flask to start
    time.sleep(2)
    
    # Start the simulation (this will block in main thread for pygame)
    if hasattr(traffic_sim, 'Main'):
        if hasattr(traffic_sim.Main, 'run'):
            traffic_sim.Main.run()  # If it's a static method
        else:
            traffic_sim.Main()  # If it's the original class
    else:
        print("❌ Could not find Main class in traffic_sim.py")

def main():
    print("\n" + "=" * 70)
    print("🚦  ML TRAFFIC CONTROL SYSTEM - UNIFIED LAUNCHER")
    print("=" * 70)
    print("\n🚀 Starting all components in ONE process...")
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask_dashboard, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to be ready
    print("\n⏳ Waiting for Flask to start...")
    time.sleep(3)
    
    # Open browser
    print("🌐 Opening dashboard in browser...")
    try:
        webbrowser.open('http://127.0.0.1:5000')
    except:
        print("⚠️  Could not auto-open browser. Please open: http://127.0.0.1:5000")
    
    print("\n" + "=" * 70)
    print("✅ SYSTEM RUNNING")
    print("=" * 70)
    print("📊 Dashboard: http://127.0.0.1:5000")
    print("🎮 Simulation: Pygame window will open...")
    print("\n💡 Everything runs in THIS terminal")
    print("🛑 Close pygame window or press CTRL+C to stop")
    print("=" * 70 + "\n")
    
    # Start simulation in main thread (blocks here)
    start_simulation()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        print("✅ Stopped")
        sys.exit(0)