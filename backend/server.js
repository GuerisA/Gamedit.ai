const express = require("express");
const multer = require("multer");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use(express.json());

// Configure Multer for file uploads
const upload = multer({ dest: "uploads/" }); // Files will be saved in the 'uploads' directory

// Route for handling video uploads
app.post("/upload", upload.single("video"), async (req, res) => {
    try {
        console.log("Uploaded File:", req.file);

        if (!req.file) {
            return res.status(400).send({ error: "No file uploaded" });
        }

        const videoPath = req.file.path; // Path of the uploaded file

        // Forward the file to the AI service
        const formData = new FormData();
        formData.append("video", fs.createReadStream(videoPath));

        const aiResponse = await axios.post("http://127.0.0.1:5001/process", formData, {
            headers: formData.getHeaders(),
        });

        // Respond with the AI service's response
        res.json(aiResponse.data);
    } catch (error) {
        console.error("Error processing video:", error.message);
        res.status(500).send("Error processing video");
    }
});

// Start the server
app.listen(5000, () => console.log("Backend server running on http://localhost:5000"));

app.get("/", (req, res) => {
    res.send("Backend server is running!");
});
