import { spawn, type ChildProcessWithoutNullStreams } from 'child_process';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_MODEL_SCRIPT = path.resolve(__dirname, '..', 'ml', 'pothole_model.py');

export interface PythonInferenceRequest {
  imageBase64: string;
  confidence?: number;
  overlap?: number;
}

export interface PythonInferenceResponse {
  inferenceId: string;
  inferenceMs: number;
  modelId: string;
  engineRuntime: string;
  pythonVersion: string;
  image: { width: number; height: number };
  predictions: any[];
}

interface PendingJob {
  resolve: (value: PythonInferenceResponse) => void;
  reject: (reason: Error) => void;
  timer: NodeJS.Timeout;
}

/**
 * Maintains a warm, persistent Python 3 worker process (`ml/pothole_model.py --worker`)
 * communicating over NDJSON stdio so every video/camera frame runs through the Python
 * model with near-zero startup overhead and automatic crash recovery.
 */
class PythonPotholeModelBridge {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private readyPromise: Promise<void> | null = null;
  private pending = new Map<string, PendingJob>();
  private nextId = 1;

  private ensureWorker(): Promise<void> {
    if (this.proc && !this.proc.killed && this.readyPromise) {
      return this.readyPromise;
    }

    this.readyPromise = new Promise<void>((resolveReady, rejectReady) => {
      let settled = false;
      const proc = spawn('python3', ['-u', PYTHON_MODEL_SCRIPT, '--worker'], {
        env: { ...process.env },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.proc = proc;

      const startupTimeout = setTimeout(() => {
        if (!settled) {
          settled = true;
          rejectReady(new Error('Python pothole model worker timed out during startup.'));
        }
      }, 6000);

      const rl = readline.createInterface({ input: proc.stdout });
      rl.on('line', (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        let msg: any;
        try {
          msg = JSON.parse(trimmed);
        } catch {
          return;
        }

        if (msg.ready) {
          if (!settled) {
            settled = true;
            clearTimeout(startupTimeout);
            resolveReady();
          }
          return;
        }

        const id = String(msg.id ?? '');
        const job = this.pending.get(id);
        if (!job) return;
        this.pending.delete(id);
        clearTimeout(job.timer);

        if (msg.ok) {
          job.resolve(msg.result as PythonInferenceResponse);
        } else {
          job.reject(new Error(msg.error || 'Python model inference failed.'));
        }
      });

      proc.stderr.on('data', (chunk) => {
        const text = chunk.toString().trim();
        if (text) {
          console.error('[python-pothole-model]', text);
        }
      });

      const handleExit = (errMsg: string) => {
        clearTimeout(startupTimeout);
        if (!settled) {
          settled = true;
          rejectReady(new Error(errMsg));
        }
        if (this.proc === proc) {
          this.proc = null;
          this.readyPromise = null;
        }
        for (const [id, job] of this.pending.entries()) {
          clearTimeout(job.timer);
          job.reject(new Error(errMsg));
          this.pending.delete(id);
        }
      };

      proc.on('error', (err) => {
        handleExit(`Python model worker error: ${err.message}`);
      });

      proc.on('exit', (code, signal) => {
        handleExit(
          `Python model worker exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`
        );
      });
    });

    return this.readyPromise;
  }

  public async infer(
    request: PythonInferenceRequest
  ): Promise<PythonInferenceResponse> {
    await this.ensureWorker();
    if (!this.proc || this.proc.killed || !this.proc.stdin.writable) {
      throw new Error('Python model worker stdin is not writable.');
    }

    const id = `py_${this.nextId++}`;
    return new Promise<PythonInferenceResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('Python pothole model inference timed out (18s).'));
      }, 18000);

      this.pending.set(id, { resolve, reject, timer });
      const payload = JSON.stringify({
        id,
        imageBase64: request.imageBase64,
        confidence: request.confidence ?? 0.28,
        overlap: request.overlap ?? 0.45,
      });
      this.proc!.stdin.write(`${payload}\n`);
    });
  }

  public shutdown(): void {
    if (this.proc && !this.proc.killed) {
      this.proc.kill();
    }
    this.proc = null;
    this.readyPromise = null;
  }
}

const bridge = new PythonPotholeModelBridge();

export function runPythonPotholeInference(
  request: PythonInferenceRequest
): Promise<PythonInferenceResponse> {
  return bridge.infer(request);
}

export function shutdownPythonPotholeWorker(): void {
  bridge.shutdown();
}
