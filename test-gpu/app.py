from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/gpu-status')
def gpu_status():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return jsonify({
        "gpu_available": gpu_available,
        "gpu_name": gpu_name
    })

@app.route('/test-gpu')
def test_gpu():
    try:
        if not torch.cuda.is_available():
            return jsonify({"success": False, "error": "CUDA not available"})
        
        # Simple GPU operation
        x = torch.rand(10000, 10000).cuda()
        y = torch.mm(x, x)
        return jsonify({"success": True, "result_sum": y.sum().item()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
