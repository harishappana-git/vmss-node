const { randomUUID } = require('crypto');

class InMemoryTodoStore {
  constructor () {
    this.todos = new Map();
  }

  list () {
    return Array.from(this.todos.values());
  }

  create ({ title }) {
    if (typeof title !== 'string' || title.trim() === '') {
      const err = new Error('title is required');
      err.code = 'VALIDATION_ERROR';
      throw err;
    }

    const todo = {
      id: randomUUID(),
      title: title.trim(),
      done: false,
      createdAt: new Date().toISOString()
    };

    this.todos.set(todo.id, todo);
    return todo;
  }
}

module.exports = {
  InMemoryTodoStore
};
