// Function to update the slider value
function updateSliderValue(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueId);

    slider.addEventListener('input', function() {
        valueDisplay.textContent = slider.value; // Update the display value
    });
}

// Call the function for each slider
updateSliderValue("snr", "snr-value");
updateSliderValue("freq", "freq-value");
updateSliderValue("amplitude", "amplitude-value");
updateSliderValue("phase", "phase-value");

// Classify signal function (existing code)
async function classifySignal() {
    // Get the input values from the form
    const snr = document.getElementById("snr").value;
    const freq = document.getElementById("freq").value;
    const amplitude = document.getElementById("amplitude").value;
    const phase = document.getElementById("phase").value;

    // Show the loader while the signal is being generated
    document.querySelector('.output').innerHTML = '<div class="loader"></div>'; // Display the loader

    // Add a small delay to ensure loader animation is visible
    setTimeout(async () => {
        try {
            // Send the input data to the server
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ snr, freq, amplitude, phase }),
            });

            // Parse the response from the server
            const data = await response.json();

            // Check for any errors in the response
            if (data.error) {
                document.getElementById("prediction-text").innerText = "Error: " + data.error;
            } else {
                // Hide the loader and display the result
                document.querySelector('.output').innerHTML = `
                    <img src="data:image/png;base64,${data.plot}" alt="Generated Signal">
                    <div id="prediction-text">${data.prediction}</div>
                `;
            }
        } catch (error) {
            // In case of an error, show a relevant message
            console.error("Error:", error);
            document.querySelector('.output').innerHTML = '<p>Error while generating the signal. Please try again.</p>';
        }
    }, 1200); // Delay of 500ms (adjust the delay time as needed)
}
