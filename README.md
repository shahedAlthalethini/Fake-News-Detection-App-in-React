# Fake News Detection App and Natural Language Processing 

A web application built with React and a Python NLP backend to classify news articles as "Real" or "Fake". Users can paste the text of a news article, and the machine learning model will provide a prediction on its authenticity.


<img width="1410" height="804" alt="image" src="https://github.com/user-attachments/assets/392a4c7d-19e3-450e-8b3b-436d52ff3a78" />


## Features 

-   **Real-time Analysis:** Get instant predictions by pasting news text into the input field.
-   **Confidence Score:** View the model's confidence in its prediction (e.g., "95% Likely to be Fake").
-   **Responsive UI:** A clean and modern user interface that works on both desktop and mobile devices.
-   **RESTful API:** The React frontend communicates with a Python backend that serves the trained NLP model.

## Technology Stack 

-   **Frontend:**
    -   React.js (using Create React App)
    -   Axios (for API requests)
    -   CSS / Styled-Components (for styling)
-   **Backend (NLP API):**
    -   Python
    -   Flask / FastAPI (for creating the API)
    -   Scikit-learn / TensorFlow / PyTorch (for the machine learning model)
    -   Pandas & NumPy (for data manipulation)
    -   NLTK / SpaCy (for text preprocessing)

## How It Works 

The application follows a simple client-server architecture:

1.  The user enters news text into the **React frontend** and clicks "Analyze".
2.  An API request is sent from the frontend to the **Python backend**.
3.  The backend API receives the text, preprocesses it (e.g., tokenization, stop-word removal), and feeds it into the pre-trained NLP model.
4.  The model returns a prediction (Real/Fake) and a confidence score.
5.  The backend sends this prediction back to the React frontend.
6.  The result is displayed to the user in the UI.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Node.js and npm (or yarn) installed
-   Python 3.8+ and pip installed
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Setup the Backend (NLP API):**
    Navigate to the backend directory, create a virtual environment, and install the required Python packages.
    ```sh
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    Now, run the backend server:
    ```sh
    python app.py
    ```
    The API server should now be running on `http://localhost:5000` (or your configured port).

3.  **Setup the Frontend (React App):**
    Open a new terminal, navigate to the frontend directory, and install the npm packages.
    ```sh
    cd frontend
    npm install
    ```
    Once the installation is complete, you can run the app.

---

## Available Scripts (Frontend)

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app). In the `frontend` directory, you can run:

### `npm start`

Runs the app in development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes. You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode. See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder. It correctly bundles React in production mode and optimizes the build for the best performance. Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project and copy all configuration files directly into your project so you have full control over them.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).
