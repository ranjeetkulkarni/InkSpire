# InkSpire - Sketch to Image with GANs

InkSpire is a web application that leverages Generative Adversarial Networks (GANs) to convert sketches into colored images in real-time. The app allows users to draw sketches on a canvas, and the sketches are sent to a Flask backend for colorization using a pre-trained GAN model.

Features
Real-time Sketch-to-Image Translation: Convert sketches into colored images instantly using GANs.
Interactive Canvas: Draw sketches directly on the web interface and see the results in real-time.
Flask Backend: Handles sketch processing and utilizes a GAN model for image generation.
Requirements
Before you start, ensure you have the following installed:

Python 3.x
Flask
TensorFlow or PyTorch (depending on which framework your GAN model uses)
HTML, CSS, and JavaScript (for the frontend)
Python Libraries
To install the required Python libraries, run:

bash
Copy code
pip install -r requirements.txt
Installation
1. Clone the Repository
Clone the project to your local machine:

bash
Copy code
git clone https://github.com/your-username/inkspire.git
cd inkspire
2. Install Dependencies
Ensure that all dependencies are installed by running:

bash
Copy code
pip install -r requirements.txt
3. Set Up the Flask Backend
The backend is built using Flask. You can run the Flask server with the following command:

bash
Copy code
python app.py
This will start the backend server at http://localhost:5000.

4. Frontend Setup
The frontend consists of HTML, CSS, and JavaScript to create an interactive canvas. The canvas allows users to draw sketches, which are then sent to the Flask backend for colorization.

To run the frontend:

Open index.html in your browser.
Draw on the canvas.
The sketch will be sent to the backend for processing every second, and the result will be displayed.
5. Running the App
Once the backend server is running and you have the frontend files, open index.html in a browser. Start drawing on the canvas, and the colorized image will appear in real-time.

How It Works
Frontend: The user draws on an HTML5 canvas. JavaScript sends the canvas data to the Flask backend every second.
Backend (Flask): The Flask app processes the incoming sketches and sends them to the GAN model for colorization.
GAN Model: A pre-trained GAN model takes the sketch and generates a colored image in real-time, which is then sent back to the frontend for display.
Technologies Used
Generative Adversarial Networks (GANs): For sketch-to-image translation.
Flask: Python web framework for backend.
TensorFlow/PyTorch: For implementing and running the GAN model.
HTML5 Canvas: For drawing sketches in the frontend.
JavaScript: For handling real-time interaction with the backend.
Screenshots

Above: Screenshot of the interactive canvas where users can draw sketches.


Above: Example of a sketch after colorization by the GAN model.

Contributing
If you'd like to contribute to this project, feel free to open an issue or create a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
