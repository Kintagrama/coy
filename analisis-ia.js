document.getElementById('input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Mostrar carga
    const chatFrame = document.querySelector('.Frame4');
    chatFrame.innerHTML = '<div style="color:white">Analizando imagen...</div>';

    try {
        // 1. Crear FormData para enviar la imagen
        const formData = new FormData();
        formData.append('image', file);

        // 2. Enviar a Flask
      const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        // 3. Procesar respuesta
        if (!response.ok) throw new Error('Error en el análisis');
        const result = await response.json();

        // 4. Mostrar resultados en tu HTML
        chatFrame.innerHTML = `
            <div style="color:white">
                <h3>Diagnóstico: ${result.diagnosis}</h3>
                <p>Precisión: ${(result.confidence * 100).toFixed(1)}%</p>
                <ul>
                    ${result.treatment.map(t => `<li>${t}</li>`).join('')}
                </ul>
            </div>
        `;

    } catch (error) {
        chatFrame.innerHTML = `<div style="color:red">Error: ${error.message}</div>`;
    }
});
