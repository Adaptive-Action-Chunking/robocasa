import zmq
import time
from policy_services import TorchSerializer 

def run_server(host="*", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")

    print(f"[SERVER] Running on {host}:{port}...")

    try:
        while True:
            message = socket.recv()
            try:
                request = TorchSerializer.from_bytes(message)
                # print("request:", request)
                endpoint = request.get("endpoint")
                data = request.get("data", None)

                print(f"[SERVER] Received request: {endpoint}")

                if endpoint == "ping":
                    socket.send(TorchSerializer.to_bytes({"status": "ok"}))

                elif endpoint == "kill":
                    print("[SERVER] Kill command received. Shutting down.")
                    socket.send(TorchSerializer.to_bytes({"status": "server shutting down"}))
                    break

                elif endpoint == "get_action":
                    # 这里返回一个假动作数据，模拟动作决策
                    dummy_action = {
                        "move": [1.0, 0.0, 0.0],
                        "gripper": "close"
                    }
                    socket.send(TorchSerializer.to_bytes(dummy_action))

                else:
                    print("[SERVER] Unknown endpoint.")
                    socket.send(b"ERROR")

            except Exception as e:
                print(f"[SERVER] Exception occurred: {e}")
                socket.send(b"ERROR")

    finally:
        socket.close()
        context.term()
        print("[SERVER] Server terminated.")

if __name__ == "__main__":
    run_server()
