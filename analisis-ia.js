

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
    <div style="color:white; font-family: Arial, sans-serif;">
        <h2 style="color: #FFC42B;">🔍 Resultado del Análisis</h2>
        <p style="font-size: 18px; margin-bottom: 20px;">
            Se ha detectado: <strong>"${result.predictions[0].label}"</strong>
        </p>
        <div style="background: #2A2A2A; padding: 15px; border-radius: 8px;">
            <h3 style="margin-top: 0;">Detalles:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                               ${result.predictions.slice(1).map(pred => `
                    <li style="margin-bottom: 8px;">
                        ▸ ${pred.label}: ${pred.confidence}
                    </li>
                `).join('')}
            </ul>
        </div>
    </div>
`;
    } catch (error) {
        chatFrame.innerHTML = `<div style="color:red">Error: ${error.message}</div>`;
    }
});
