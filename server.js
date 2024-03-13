const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const multer = require('multer');

const app = express();
const PORT = 3000;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

const storage = multer.diskStorage({
    destination(req, file, done) {
        done(null, 'uploads/');
    },

    filename(req, file, done) {
        const ext = path.extname(file.originalname);
        const fileName = `${path.basename(file.originalname, ext)}_${Date.now()}${ext}`;
        done(null, fileName);
    },
});

const limits = { fileSize: 5 * 1024 * 1024 };

const multerConfig = {
    storage,
    limits,
};

const upload = multer(multerConfig);

app.use('/uploads', express.static('uploads'));

app.post('/save/image', upload.single('profile'), (req, res) => {
    const imageUrl = `/uploads/${req.file.filename}`;
    res.status(200).json({ message: 'image save success', imageUrl });
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
