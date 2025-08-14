const express = require('express');
const http = require('http');
const cors = require('cors');
const { Server } = require('socket.io');
const CryptoJS = require('crypto-js');

const SECRET_KEY = process.env.SOCKET_SECRET || 'dev-secret';

const app = express();
app.use(cors());

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: '*', 
    methods: ['GET', 'POST']
  }
});

const PORT = process.env.PORT || 5000;


const { registerPollEvents } = require('./pollManager');


function calculateEntropy(distribution) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let entropy = 0;
  for (const count of Object.values(distribution)) {
    const p = count / total;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}


io.use((socket, next) => {
  try {
    const { token, role } = socket.handshake.auth || {};

    if (!token || !role) {
      return next(new Error('Missing token or role'));
    }


    let decrypted;
    try {
      const bytes = CryptoJS.AES.decrypt(token, SECRET_KEY);
      decrypted = bytes.toString(CryptoJS.enc.Utf8);
    } catch {
      return next(new Error('Invalid token format'));
    }

    if (!decrypted || !/^\d{13}-[0-9a-f]+$/.test(decrypted)) {
      return next(new Error('Token validation failed'));
    }

    socket.data.role = role;
    socket.data.token = decrypted;
    next();
  } catch {
    next(new Error('Auth middleware error'));
  }
});

io.on('connection', (socket) => {
  console.log(`✅ ${socket.data.role} connected: ${socket.id}`);

  // Room joiners
  socket.on('join-room', ({ roomId }) => {
    if (roomId) {
      socket.join(roomId);
      console.log(`${socket.data.role} joined room ${roomId}`);
    }
  });

  socket.on('join-teacher', ({ roomId }) => {
    if (socket.data.role === 'teacher' && roomId) {
      socket.join(roomId);
      console.log(`Teacher joined room ${roomId}`);
    }
  });

  socket.on('join-student', ({ roomId }) => {
    if (socket.data.role === 'student' && roomId) {
      socket.join(roomId);
      console.log(`Student joined room ${roomId}`);
    }
  });
  registerPollEvents(io, socket);
});

server.listen(PORT, () =>
  console.log(`🚀 Server running on port ${PORT}`)
);
