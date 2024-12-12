const express = require('express');
const mysql = require('mysql');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

// MySQL ���� ����
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'root',
    database: 'driverdata'
});

// MySQL ����
connection.connect();

// Body parser ����
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// ���Ʈ ����
app.get('/sleepdata', (req, res) => {
    const query = 'SELECT * FROM sleepdetection';
    connection.query(query, (error, results) => {
        if (error) throw error;
        res.json(results);
    });
});

// ���� ����
app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});
