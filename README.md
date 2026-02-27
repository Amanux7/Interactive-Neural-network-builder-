# Interactive Neural Network Builder (NeuralForge)

A modern, interactive, web-based visual editor for designing, simulating, and exporting Neural Network architectures. Built with React and tailored for both educational exploration and rapid prototyping.

## 🚀 Features

*   **Drag-and-Drop Canvas:** Visually construct neural networks by dragging layer nodes (Input, Dense, Conv2D, LSTM, Embedding, Dropout, etc.) onto the canvas and wiring their connections.
*   **Real-time Training Simulation:** Hit 'Space' to watch simulated forward propagation and gradient descent. Includes live epoch tracking, loss/accuracy readouts, and visually pulsing weight updates.
*   **Summary Dashboard:** A pop-out metrics panel showing live accuracy and loss dials, total trainable parameter estimations, and a sparkline chart of training history.
*   **Pre-Built Templates:** Instantly load complete architectures like an MNIST classifier, a Binary Classifier, or a standard ConvNet.
*   **TensorFlow.js Export:** Download your architecture as a valid JSON file detailing layer configurations, parameter counts, training history, and a JavaScript boilerplate snippet for running the model in TF.js.
*   **Interactive Tutorial:** A step-by-step visual spotlight tour that teaches new users how to use the builder, wire nodes, and export models.
*   **AI Optimizer:** Analyzes your architecture and suggests improvements (e.g., adding Dropout to prevent overfitting, switching activation functions, or normalizing inputs) running completely legally via WebLLM.
*   **Live Collaboration:** Team up with others using a built-in mock collaborative engine representing real-time cursors and presence.

## 🛠️ Technology Stack

*   **Frontend Framework:** React 18 + TypeScript
*   **Build Tool:** Vite
*   **Styling:** Tailwind CSS (with advanced custom animations and dark mode support)
*   **Icons:** Lucide React
*   **Charts:** Recharts
*   **Local AI:** WebLLM for the architecture optimizer suggestions

## 📦 Getting Started

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

1. Clone or download this repository.
2. Navigate to the project directory in your terminal:
   ```bash
   cd "Interactive neural network builder"
   ```
3. Install the dependencies:
   ```bash
   npm install
   ```

### Running the Application

Start the development server:
```bash
npm run dev
```
The application will be available at `http://localhost:5173`.

### Building for Production

To create a production-ready build:
```bash
npm run build
```
The compiled assets will be placed in the `dist/` directory.

## 🎨 Application Previews
- **Dark & Light Mode:** Fully cohesive color systems switching natively.
- **Glassmorphism:** Elegant contextual menus, overlays, and sidebars.
- **Micro-interactions:** Animated bezier connections traversing data particles to visualize the training loop.