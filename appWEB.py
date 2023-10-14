from flask import Flask, render_template, request
import knn_module

app = Flask(__name__)

# Treine o modelo e obtenha a acurácia quando o aplicativo é iniciado
knn, ct, X_test, y_test = knn_module.train_knn(k=20)
accuracy = knn_module.evaluate_model(knn, X_test, y_test)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    highlighted_point = None
    if request.method == 'POST':
        idade = float(request.form.get('idade'))
        genero = request.form.get('genero')
        nivel_gasto = float(request.form.get('nivel_gasto'))
        tempo_site = float(request.form.get('tempo_site'))
        sample = [idade, genero, nivel_gasto, tempo_site]
        prediction = knn_module.predict_sample(knn, ct, sample)
        highlighted_point = (idade, nivel_gasto)

    plotly_html = knn_module.generate_plotly_figure(knn, ct, highlighted_point=highlighted_point)
    return render_template('index.html', prediction=prediction, plotly_html=plotly_html, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)