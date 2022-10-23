import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import rossmann

# instanciando as requisições
app = Flask(__name__)

# puchando meu modelo em memória
model = pickle.load(open('/users/leona/Comunidade_DS/repos/proj2_ds_em_prod/model/model_do_rossmann.pkl', 'rb'))


# ativando o endpoint
@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    # instanciando o json vindo da api
    test_json = request.get_json()

    # se tiver dados json
    if test_json:
        # se for uma única linha json
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index[0])
        # se for mais de uma linha json
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instanciando minha classe
        pepiline = rossmann()

        # radando meu método de limpeza
        df1 = pepiline.cleaning(test_raw)

        # rodando meu método de feature eng
        df2 = pepiline.feature_eng(df1)

        # rodando meu método de preparação
        df3 = pepiline.preparation(df2)

        # rodando o modelo para predição
        df_response = pepiline.prediction(model, test_raw, df3)

        return df_response

    # se não haver dados na api
    else:
        return Response('{}', status=200, mimetype='application/json')


# primeiro a rodas
if __name__ == '__main__':
    app.run('0.0.0.0')