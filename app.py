import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up the page configuration
st.set_page_config(page_title="Spaceship Titanic Prediction App")

# Define containers
about_container = st.container()
train_container = st.container()
test_container = st.container()

# Sidebar for navigation
with st.sidebar:
    st.write("## Navigation")
    page = st.radio("Choose a page:", ["About", "Train", "Test"])

# About Container
with about_container:
    if page == "About":
        st.header("About the App")
        st.write("This app predicts the 'Transported' status of passengers on the Spaceship Titanic.")

# Train Container
with train_container:
    if page == "Train":
        st.header("Train the Model")
        uploaded_file = st.file_uploader("Upload Training CSV", type="csv", key="train_uploader")

        if uploaded_file is not None:
            train_df = pd.read_csv(uploaded_file)

            # Button to train the model
            if st.button('Train Model'):
                # Define columns
                numerical_cols = [cname for cname in train_df.columns if 
                                  train_df[cname].dtype in ['int64', 'float64'] and 
                                  cname not in ['Age']]
                categorical_cols = [cname for cname in train_df.columns if
                                    train_df[cname].nunique() < 10 and 
                                    train_df[cname].dtype == "object" and 
                                    cname not in ['HomePlanet', 'Destination']]

                # Define transformers
                numerical_transformer = SimpleImputer(strategy='mean')
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                # Preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, numerical_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])

                # Model
                model = RandomForestClassifier(n_estimators=100, random_state=0)
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

                # Prepare data
                X = train_df.drop(['Transported', 'PassengerId', 'Name', 'Cabin', 'Destination', 'HomePlanet', 'Age'], axis=1)
                y = train_df['Transported']
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

                # Train the model and calculate accuracy
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_valid, y_valid)

                # Save the trained model
                with open('model.pkl', 'wb') as file:
                    pickle.dump(clf, file)

                st.write("Training Dataset Information:")
                st.write(f"Number of Rows: {train_df.shape[0]}")
                st.write(f"Number of Columns: {train_df.shape[1]}")
                st.write(f"List of Columns: {list(train_df.columns)}")
                st.write(f"Trained Model Accuracy: {accuracy:.4f}")
                st.success("Model trained and saved successfully.")


# Test Container
with test_container:
    if page == "Test":
        st.header("Test the Model")
        uploaded_file = st.file_uploader("Upload Test CSV", type="csv", key="test_uploader")

        if uploaded_file is not None:
            test_df = pd.read_csv(uploaded_file)
            st.write("Test Dataset Uploaded. Ready to make predictions.")

            # Button to test the model
            if st.button('Test Model'):
                # Load the trained model
                with open('model.pkl', 'rb') as file:
                    model = pickle.load(file)

                # Prepare test data
                X_test = test_df.drop(['PassengerId', 'Name', 'Cabin', 'Destination', 'HomePlanet', 'Age'], axis=1)

                # Make predictions
                preds_test = model.predict(X_test)

                # Prepare and display the output
                output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds_test})
                st.dataframe(output)

                # Download link for predictions
                st.download_button(
                    label="Download Predictions as CSV",
                    data=output.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv',
                )


            #streamlit run app.py
