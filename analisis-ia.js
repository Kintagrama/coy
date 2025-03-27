const skinTypeTranslations = {
    'oily': 'piel grasa',
    'dry': 'piel eca',
    'combination': 'piel mixta',
    'normal': 'piel normal',
    'acne': 'acné',
    'rosacea': 'rosácea'
};

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
            Se ha detectado piel <strong>"${skinTypeTranslations[result.predictions[0].label.toLowerCase()] || result.predictions[0].label}"</strong>
        </p>
        <div style="background: #2A2A2A; padding: 15px; border-radius: 8px;">
            <h3 style="margin-top: 0;">Detalles:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                ${result.predictions.map(pred => {
                    const translatedLabel = skinTypeTranslations[pred.label.toLowerCase()] || pred.label;
                    return `
                    <li style="margin-bottom: 8px;">
                        ▸ ${translatedLabel}: <span style="color: #4CAF50;">${pred.confidence}</span>
                    </li>`;
                }).join('')}
            </ul>
        </div>
    </div>

    } catch (error) {
        chatFrame.innerHTML = `<div style="color:red">Error: ${error.message}</div>`;
    }
});
