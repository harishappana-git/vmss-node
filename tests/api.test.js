const request = require('supertest');
const { createServer } = require('../src/server');

describe('API', () => {
  let app;

  beforeAll(() => {
    process.env.GIT_SHA = 'test-sha';
    app = createServer();
  });

  test('GET /healthz returns ok with commit sha', async () => {
    const response = await request(app).get('/healthz');
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ status: 'ok', commit: 'test-sha' });
  });

  test('POST /api/v1/todos creates a todo and GET lists it', async () => {
    const createResponse = await request(app)
      .post('/api/v1/todos')
      .send({ title: 'first todo' })
      .set('Content-Type', 'application/json');

    expect(createResponse.status).toBe(201);
    expect(createResponse.body.data).toMatchObject({
      title: 'first todo',
      done: false
    });

    const listResponse = await request(app).get('/api/v1/todos');
    expect(listResponse.status).toBe(200);
    expect(Array.isArray(listResponse.body.data)).toBe(true);
    expect(listResponse.body.data.length).toBeGreaterThanOrEqual(1);
  });

  test('POST /api/v1/todos validation failure returns 400', async () => {
    const response = await request(app)
      .post('/api/v1/todos')
      .send({ title: '' })
      .set('Content-Type', 'application/json');

    expect(response.status).toBe(400);
    expect(response.body).toEqual({ error: 'title is required' });
  });
});
