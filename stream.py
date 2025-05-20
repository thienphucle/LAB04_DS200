import socket, time, pickle, argparse, os, json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', type=str, required=True)
parser.add_argument('--batch-size', '-b', type=int, required=True)
parser.add_argument('--split', '-s', type=str, default="train")
parser.add_argument('--sleep', '-t', type=int, default=1)
args = parser.parse_args()

TCP_IP = "localhost"
TCP_PORT = 8888

def load_batches(folder, split):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(split)]

def send_batches(sock, files, batch_size):
    for file in files:
        with open(file, "rb") as f:
            batch = pickle.load(f)
        data = np.array(batch[b'data'])
        labels = batch[b'labels']
        for i in range(0, len(data), batch_size):
            images = data[i:i+batch_size]
            lbls = labels[i:i+batch_size]
            payload = {}
            for idx, (img, lbl) in enumerate(zip(images, lbls)):
                sample = {f"feature-{j}": float(v) for j, v in enumerate(img)}
                sample["label"] = int(lbl)
                payload[idx] = sample
            sock.send((json.dumps(payload) + "\n").encode())
            time.sleep(args.sleep)

if __name__ == "__main__":
    s = socket.socket()
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    print(f"Listening on port {TCP_PORT}...")
    conn, _ = s.accept()
    print("Connected.")

    files = load_batches(args.folder, args.split)
    send_batches(conn, files, args.batch_size)

    conn.close()
