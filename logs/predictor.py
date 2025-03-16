import pandas as pd
import joblib
import logging
import numbers
from typing import Union, List, Dict

# Configure logging (or use your project's logging configuration)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class LoanDataTransformer:
    """
    Transforms raw loan application data by applying:
      - Explicit mapping on previous_loan_defaults_on_file (case-sensitive: "No" -> 0, "Yes" -> 1)
      - One-hot encoding on person_home_ownership and loan_intent with fixed output columns (0 and 1)
      - Scaling on numeric columns using a pre-saved scaler.
      
    Validates input and logs warnings for extra keys.
    """
    
    def __init__(self, scaler_path: str):
        try:
            self.scaler = joblib.load(scaler_path)
            logging.info("Scaler loaded successfully from %s.", scaler_path)
        except Exception as e:
            logging.error("Failed to load scaler from %s: %s", scaler_path, str(e))
            raise
        
        # Explicit mapping for previous_loan_defaults_on_file.
        self.defaults_mapping = {"No": 0, "Yes": 1}
        
        # Define expected one-hot encoded column names.
        self.expected_home_columns = [
            "person_home_ownership_MORTGAGE", 
            "person_home_ownership_OTHER", 
            "person_home_ownership_OWN", 
            "person_home_ownership_RENT"
        ]
        self.expected_intent_columns = [
            "loan_intent_DEBTCONSOLIDATION", 
            "loan_intent_EDUCATION", 
            "loan_intent_HOMEIMPROVEMENT", 
            "loan_intent_MEDICAL", 
            "loan_intent_PERSONAL", 
            "loan_intent_VENTURE"
        ]
        self.final_columns = (
            ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'] +
            ['previous_loan_defaults_on_file'] +
            self.expected_home_columns +
            self.expected_intent_columns
        )
        
        self.expected_input_keys = {
            "person_income", "person_home_ownership", "loan_amnt", 
            "loan_intent", "loan_int_rate", "loan_percent_income", 
            "previous_loan_defaults_on_file"
        }
    
    def _validate_input(self, record: Dict):
        for key in self.expected_input_keys:
            if key not in record:
                raise ValueError(f"Missing required field: {key}")
        
        # Validate numeric fields
        for key in ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']:
            if not isinstance(record[key], numbers.Number):
                raise ValueError(f"Field '{key}' must be numeric. Got {type(record[key])} instead.")
        
        # Validate string fields
        for key in ['person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']:
            if not isinstance(record[key], str):
                raise ValueError(f"Field '{key}' must be a string. Got {type(record[key])} instead.")
        
        # Validate and auto-correct categorical fields
        valid_home_ownerships = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if record["person_home_ownership"] not in valid_home_ownerships:
            logging.warning(
                f"Invalid person_home_ownership value: '{record['person_home_ownership']}'. "
                f"Auto-converting to 'OTHER'."
            )
            record["person_home_ownership"] = "OTHER"
        
        valid_loan_intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        if record["loan_intent"] not in valid_loan_intents:
            logging.warning(
                f"Invalid loan_intent value: '{record['loan_intent']}'. "
                f"Auto-converting to 'PERSONAL'."
            )
            record["loan_intent"] = "PERSONAL"
        
        # Validate previous loan defaults field (case-sensitive)
        if record["previous_loan_defaults_on_file"] not in self.defaults_mapping:
            raise ValueError("Field 'previous_loan_defaults_on_file' must be 'No' or 'Yes'.")
    
    def transform(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        try:
            if isinstance(data, dict):
                data = [data]
            for record in data:
                self._validate_input(record)
            all_keys = set().union(*(record.keys() for record in data))
            extra_keys = all_keys - self.expected_input_keys
            if extra_keys:
                logging.warning("The following columns are provided but not used: %s", list(extra_keys))
            df = pd.DataFrame(data)
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(self.defaults_mapping)
            df_home = pd.get_dummies(df['person_home_ownership'], prefix='person_home_ownership').astype(int)
            df_home = df_home.reindex(columns=self.expected_home_columns, fill_value=0)
            df_intent = pd.get_dummies(df['loan_intent'], prefix='loan_intent').astype(int)
            df_intent = df_intent.reindex(columns=self.expected_intent_columns, fill_value=0)
            numeric_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
            df_numeric = df[numeric_cols]
            df_scaled = pd.DataFrame(
                self.scaler.transform(df_numeric),
                columns=numeric_cols,
                index=df.index
            )
            df_label = df[['previous_loan_defaults_on_file']]
            df_transformed = pd.concat([df_scaled, df_label, df_home, df_intent], axis=1)
            df_transformed = df_transformed.reindex(columns=self.final_columns, fill_value=0)
            return df_transformed
        except Exception as e:
            logging.error("Error during data transformation: %s", str(e))
            raise

class LoanApprovalPredictor:
    """
    Loads a pre-trained model and uses the data transformer to preprocess input data,
    then makes predictions regarding loan approval status.
    
    Model choice can be specified among:
      - "Decision_Tree"
      - "KNN"
      - "Logistic_Regression"
      - "Random_Forest"
      
    If an invalid model is provided, defaults to KNN.
    """
    
    def __init__(self, scaler_path: str, model_choice: str = "KNN"):
        allowed_models = {
            "Decision_Tree": "logs/model_files/Decision_Tree.pkl",
            "KNN": "logs/model_files/KNN.pkl",
            "Logistic_Regression": "logs/model_files/Logistic_Regression.pkl",
            "Random_Forest": "logs/model_files/Random_Forest.pkl"
        }
        if model_choice not in allowed_models:
            logging.warning("Model choice '%s' not recognized. Defaulting to KNN model.", model_choice)
            model_choice = "KNN"
        model_path = allowed_models[model_choice]
        self.transformer = LoanDataTransformer(scaler_path)
        try:
            self.model = joblib.load(model_path)
            logging.info("Model '%s' loaded successfully from %s.", model_choice, model_path)
        except Exception as e:
            logging.error("Failed to load model from %s: %s", model_path, str(e))
            raise
    
    def predict(self, data: Union[Dict, List[Dict]]) -> List:
        try:
            processed_data = self.transformer.transform(data)
            input_array = processed_data.values
            predictions = self.model.predict(input_array)
            return predictions.tolist()
        except Exception as e:
            logging.error("Error during prediction: %s", str(e))
            raise