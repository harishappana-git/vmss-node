const express = require('express');
const pino = require('pino');
const pinoHttp = require('pino-http');
const client = require('prom-client');

const healthRouter = require('./routes/health');
const todoRouter = require('./routes/todos');
const { InMemoryTodoStore } = require('./db/memory');

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const register = new client.Registry();
client.collectDefaultMetrics({ register });

const requestDuration = new client.Histogram({
  name: 'http_server_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 1, 2, 5]
});

const todosCreated = new client.Counter({
  name: 'app_todos_created_total',
  help: 'Total number of todos created'
});

register.registerMetric(requestDuration);
register.registerMetric(todosCreated);

function createServer () {
  const app = express();
  const todoStore = new InMemoryTodoStore();

  app.locals.todoStore = todoStore;
  app.locals.metrics = {
    requestDuration,
    todosCreated
  };

  app.use(
    pinoHttp({
      logger,
      customLogLevel: function (res, err) {
        if (res.statusCode >= 500 || err) return 'error';
        if (res.statusCode >= 400) return 'warn';
        return 'info';
      },
      customSuccessMessage: function (req, res) {
        return `${req.method} ${req.url} ${res.statusCode}`;
      },
      customErrorMessage: function (req, res, err) {
        return `${req.method} ${req.url} ${res.statusCode} - ${err.message}`;
      }
    })
  );

  app.use(express.json({ limit: '100kb' }));

  app.use((req, res, next) => {
    const end = requestDuration.startTimer({ method: req.method });
    res.on('finish', () => {
      const route = req.route ? `${req.baseUrl}${req.route.path}` : req.originalUrl.split('?')[0];
      end({ route, status_code: res.statusCode });
    });
    next();
  });

  app.use(healthRouter);
  app.use(todoRouter);

  app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  });

  app.use((err, req, res, next) => {
    if (err.code === 'VALIDATION_ERROR') {
      return res.status(400).json({ error: err.message });
    }

    req.log.error({ err }, 'Unhandled error');
    res.status(500).json({ error: 'Internal Server Error' });
  });

  app.use((req, res) => {
    res.status(404).json({ error: 'Not Found' });
  });

  return app;
}

function start () {
  const app = createServer();
  const port = process.env.PORT || 3000;

  app.listen(port, () => {
    logger.info({ port, gitSha: process.env.GIT_SHA || 'unknown' }, 'Server listening');
  });
}

module.exports = {
  createServer,
  start,
  register
};
