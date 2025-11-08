const express = require('express');

const router = express.Router();

router.get('/api/v1/todos', (req, res) => {
  const store = req.app.locals.todoStore;
  const todos = store.list();
  res.status(200).json({ data: todos });
});

router.post('/api/v1/todos', (req, res, next) => {
  const store = req.app.locals.todoStore;

  try {
    const todo = store.create(req.body || {});
    req.app.locals.metrics.todosCreated.inc();
    res.status(201).json({ data: todo });
  } catch (error) {
    return next(error);
  }
});

module.exports = router;
