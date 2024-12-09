const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5000;

// Middleware to parse JSON requests
app.use(bodyParser.json());

// Endpoint to run all notebooks sequentially
app.post('/run-notebooks', (req, res) => {
    const notebookFolder = path.join(__dirname, '../ML/notebooks');

    // List of notebook filenames (update if necessary)
    const notebooks = [
        //'data_preprocessing.ipynb',
        'pso_implementation.ipynb',
        //'visualization.ipynb'
    ];

    // Function to execute notebooks sequentially
    const runNotebook = (index) => {
        if (index >= notebooks.length) {
            return res.json({ message: 'All notebooks executed successfully and outputs saved.' });
        }

        const notebookPath = path.join(notebookFolder, notebooks[index]);
        const command = `papermill "${notebookPath}" "${notebookPath}"`;

        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error running ${notebooks[index]}:`, stderr);
                return res.status(500).json({
                    error: `Failed to execute ${notebooks[index]}`,
                    details: stderr,
                });
            }
            
            // Log the notebook output to the VS Code terminal
            console.log(`Executed ${notebooks[index]} successfully.`);
            console.log(stdout);  // Display notebook output in VS Code terminal

            runNotebook(index + 1); // Run the next notebook
        });
    };

    // Start executing notebooks from the first one
    runNotebook(0);
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
