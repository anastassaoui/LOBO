import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os


st.set_page_config(layout="wide")


def load_data():
    if os.path.exists('data.csv'):
        return pd.read_csv('data.csv')
    else:
        st.error("data.csv not found!")
        return pd.DataFrame(columns=['X', 'Y', 'perte_charge'])

data = load_data()

if not data.empty:
    X_features = data[['perte_charge']]
    Y_targets = data[['X', 'Y']]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_targets, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
else:
    st.warning("No data available.")
    model = None

st.title("Packed Column X-Y Prediction Tool")

pressure_drop = st.slider("Enter the desired pressure drop:", min_value=0.0, value=250.0, step=1.0)

if model:
    prediction = model.predict([[pressure_drop]])
    predicted_X, predicted_Y = prediction[0]

    st.write(f"Predicted X: {predicted_X:.6f}")
    st.write(f"Predicted Y: {predicted_Y:.6f}")

    if st.button("Save Prediction to CSV"):
        if ((data['perte_charge'] == pressure_drop) & 
            (data['X'].round(6) == round(predicted_X, 6)) & 
            (data['Y'].round(6) == round(predicted_Y, 6))).any():
            st.warning("This prediction already exists.")
        else:
            new_data = pd.DataFrame({
                'X': [predicted_X],
                'Y': [predicted_Y],
                'perte_charge': [pressure_drop]
            })
            updated_data = pd.concat([data, new_data], ignore_index=True)
            updated_data.to_csv('data.csv', index=False)
            st.success("Prediction saved!")
else:
    st.error("Model is not trained.")
