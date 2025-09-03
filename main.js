import { Worker } from 'node:worker_threads';
import os from 'node:os';

const THREADS = os.cpus().length; // núcleos disponibles
console.log(`Lanzando ${THREADS} hilos...`);

for (let t = 0; t < THREADS; t++) {
  const worker = new Worker('./worker.js', { workerData: { id: t } });
  worker.on('message', (msg) => {
    if (msg.found) {
      console.log(`\n✅ Hilo ${msg.id} encontró coincidencia:`);
      console.log(`Salt: ${msg.salt}`);
      console.log(`Dirección: ${msg.addr}`);
      process.exit(0); // mata todos los hilos
    } else if (msg.progress) {
      process.stdout.write(`\rHilo ${msg.id} → ${msg.progress} intentos`);
    }
  });
}
