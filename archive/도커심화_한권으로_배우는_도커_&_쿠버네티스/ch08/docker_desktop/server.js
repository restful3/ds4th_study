const express = require('express');
const mysql = require('mysql2/promise');
const app = express();

const dbConfig = {
  host: process.env.MYSQL_HOST || 'localhost',
  user: process.env.MYSQL_USER || 'root',
  password: process.env.MYSQL_PASSWORD || 'password',
  database: process.env.MYSQL_DATABASE || 'k8sDemo'
};

async function initializeDb() {
  const connection = await mysql.createConnection({
    ...dbConfig,
    database: null
  });
  
  await connection.query(`CREATE DATABASE IF NOT EXISTS ${dbConfig.database}`);
  await connection.query(`USE ${dbConfig.database}`);
  await connection.query(`
    CREATE TABLE IF NOT EXISTS visits (
      id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);
  await connection.end();
}

// 데이터베이스 초기화
initializeDb().catch(console.error);

app.get('/', async (req, res) => {
  try {
    const connection = await mysql.createConnection(dbConfig);
    await connection.query('INSERT INTO visits (timestamp) VALUES (NOW())');
    const [rows] = await connection.query('SELECT COUNT(*) as count FROM visits');
    await connection.end();
    
    res.send(`Hello Kubernetes! You are visitor number ${rows[0].count}`);
  } catch (err) {
    console.error('Database error:', err);
    res.status(500).send('Error connecting to database');
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
