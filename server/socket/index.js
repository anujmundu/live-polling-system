const CryptoJS = require('crypto-js');
const calculateEntropy = require('../utils/calculateEntropy'); 

const SECRET_KEY = process.env.SOCKET_SECRET || 'dev-secret';

module.exports = (io) => {
  let currentQuestion = null;
  let individualResponses = {}; 
  let responseCounts = {}; 


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
      } catch (err) {
        return next(new Error('Invalid token format'));
      }

      if (!decrypted || !/^\d{13}-[0-9a-f]+$/.test(decrypted)) {
        return next(new Error('Token validation failed'));
      }

      socket.data.role = role;
      socket.data.token = decrypted;
      next();
    } catch (err) {
      next(new Error('Auth middleware error'));
    }
  });

  io.on('connection', (socket) => {
    console.log(`✅ ${socket.data.role} connected: ${socket.id}`);


    socket.on('join-student', ({ name, roomId }) => {
      if (socket.data.role !== 'student') return;
      if (!roomId || typeof name !== 'string') return;

      socket.data.name = name.trim();
      socket.join(roomId);

      io.to(roomId).emit('student-joined', socket.data.name);
      console.log(`Student ${socket.data.name} joined room ${roomId}`);
    });


    socket.on('ask-question', (payload, ack) => {
      if (socket.data.role !== 'teacher') {
        return ack?.({ success: false, message: 'Only teachers can ask questions' });
      }

      const { roomId, question, options } = payload || {};
      if (
        !roomId ||
        typeof question !== 'string' ||
        !Array.isArray(options) ||
        options.length < 2
      ) {
        return ack?.({ success: false, message: 'Invalid question payload' });
      }


      currentQuestion = {
        questionId: Date.now().toString(),
        question: question.trim(),
        options: [...new Set(options.map(o => o.trim()))].filter(Boolean),
      };
      individualResponses = {};
      responseCounts = {};

      io.to(roomId).emit('new-question', currentQuestion);
      console.log(`📢 Question sent to ${roomId}: "${currentQuestion.question}"`);

      return ack?.({ success: true, message: 'Question broadcasted' });
    });

    socket.on('submit-answer', (payload) => {
      if (socket.data.role !== 'student') return;

      const { roomId, name, answer } = payload || {};
      if (!roomId || typeof name !== 'string' || typeof answer !== 'string') return;

      individualResponses[name.trim()] = answer;

      if (!responseCounts[answer]) {
        responseCounts[answer] = 0;
      }
      responseCounts[answer] += 1;

      const entropy = calculateEntropy(responseCounts);

      io.to(roomId).emit('update-results', {
        responses: individualResponses,
        distribution: responseCounts,
        entropy,
      });

      console.log(`📝 ${name} answered "${answer}" in ${roomId}`);
    });


    socket.on('disconnect', () => {
      console.log(`❌ ${socket.data.role} disconnected: ${socket.id}`);
    });
  });
};
