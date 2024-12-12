const express = require('express');
const mysql = require('mysql');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

// MySQL 연결 설정
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'root',
    database: 'driverdata'
});

// MySQL 연결
connection.connect();

// Body parser 설정
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// 라우트 설정
app.get('/sleepdata', (req, res) => {
    const query = 'SELECT * FROM sleepdetection';
    connection.query(query, (error, results) => {
        if (error) throw error;
        res.json(results);
    });
});

// 서버 시작
app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});
