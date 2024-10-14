// Load face-api models and initialize the app
Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),  // Face detection model
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'), // Face landmark detection model
    faceapi.nets.faceRecognitionNet.loadFromUri('/models') // Face recognition model
]).then(startApp);

function startApp() {
    const imageUpload = document.getElementById('imageUpload');
    imageUpload.addEventListener('change', handleImage);
}

// Store labeled descriptors
const labeledDescriptors = [];

async function handleImage(event) {
    const img = await faceapi.bufferToImage(event.target.files[0]);
    const canvas = faceapi.createCanvasFromMedia(img);
    document.getElementById('container').append(canvas);

    const displaySize = { width: img.width, height: img.height };
    faceapi.matchDimensions(canvas, displaySize);

    const detections = await faceapi.detectAllFaces(img)
        .withFaceLandmarks()
        .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    // Draw detected faces and landmarks
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

    // Recognize and label faces
    recognizeFaces(resizedDetections, canvas, displaySize);
}

async function recognizeFaces(detections, canvas, displaySize) {
    for (const detection of detections) {
        const descriptor = detection.descriptor;

        // Find the best match if the face was previously labeled
        const bestMatch = findBestMatch(descriptor);

        if (bestMatch.label === "unknown") {
            const label = prompt("Please enter the name for this person:");
            const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(label, [descriptor]);
            labeledDescriptors.push(labeledFaceDescriptor);
        }

        // Draw the label on the canvas
        const box = detection.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.label });
        drawBox.draw(canvas);

        // Save labeled image for future recognition
        saveLabeledImage(bestMatch.label, detection);
    }
}

function findBestMatch(descriptor) {
    if (labeledDescriptors.length === 0) {
        return { label: "unknown" };
    }

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    return faceMatcher.findBestMatch(descriptor);
}

function saveLabeledImage(label, detection) {
    console.log(`Saving image with label: ${label}`);
    // Here you would save or group the image in directories based on the label
    // For example, send the image and label to your backend for storage
}
