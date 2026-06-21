import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder={}

    def load_data(self, filepath):
        print(f"Loading data from {filepath} ....")
        df = pd.read_csv(filepath)
        print(f"\n Data Loaded")
        return df
    
    def handling_missing_values(self, df):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if df['TotalCharges'].isnull().sum()>0:
            df.fillna({'TotalCharges':df['TotalCharges'].median()}, inplace=True)
            print(f"Filled Missing values")
        return df
    
    
    # Featur Engineering
    
    def feature_engineering(self, df):

        # Create new feature
        print("\nCreating new feature")

        # check tenure range
        print(f"Tenure Range:{df['tenure'].min()} to {df['tenure'].max()}")


        # Tenure groups - FIX: Handle edge cases and NaN
        # Use right=True to include right edge, and extend bins to cover all values
        df['tenure_group'] = pd.cut(df['tenure'],
                                    bins=[-1,12,24,48,73],
                                    labels = [0,1,2,3],
                                    include_lowest = True)
        
        # convert to float first (handle NaN), then to int
        df['tenure_group'] = df['tenure_group'].astype(float).astype(int)

        # avg monthly charge per tenure
        df['avg_monthly_per_tenure']= df['TotalCharges'] / (df['tenure'] + 1) # '+1' for avoid dicvision by zero

        # Replace any inf values with 0
        df['avg_monthly_per_tenure'] = df['avg_monthly_per_tenure'].replace([np.inf, -np.inf], 0)

        # Has multiple services
        service_cols = ['PhoneService', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport']

        # count services (Handling 'No internet services' properly)
        df['num_services'] = 0
        for col in service_cols:
            if col in df.columns:
                # Count as service if value is 'Yes'
                df['num_services'] += (df[col] == 'Yes').astype(int)

        
        print(f"Creted 3 new features")
        print(f" -tenure =_group:{df['tenure_group'].unique()}")
        print(f" -avg monthly per tenure: min={df['avg_monthly_per_tenure'].min():.2f}, max={df['avg_monthly_per_tenure'].max():.2f}")
        print(f"   - num_services: min={df['num_services'].min()}, max={df['num_services'].max()}")

        return df
    
    # Encoding
    def encode_features(self, df, is_training=False):
        if 'customerID' in df.columns:
            df = df.drop("customerID", axis=1)
        
        #TODO: Binary encoding for Yes/No
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df[col] = df[col].map({'Yes':1, 'No':0})

        #For Target variable 
        df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
        df['gender'] = df['gender'].map({'Male':1,'Female':0}) 

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoder[col] = le
                print(f"   - Encoded {col}: {len(le.classes_)} classes")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))

        print(f"Encoded {len(binary_cols) + len(categorical_cols) + 1} categorical features")

        return df
    

    # Feature Scaling

    def scale_features(self, X_train, X_test = None, is_training = True):
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure']
        
        numerical_cols = [col for col in numerical_cols if col in X_train.columns]

        if is_training:
            X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            if X_test is not None:
                X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        else:
            X_train[numerical_cols] = self.scaler.transform(X_train[numerical_cols])
        
        print("FEATURES SCALED!!!")
        return X_train,X_test
    

    #TODO: Preprocessing Pipeline

    def prepare_data(self, df, test_size=0.2, random_state=42):

        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        #handling missing values
        df = self.handling_missing_values(df)
        
        #feature engineering
        df = self.feature_engineering(df)
        
        #encoding
        df = self.encode_features(df, True)
        
        #split Features and Target
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # Traintest split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)

        #scale Features
        X_train, X_test = self.scale_features(X_train, X_test, is_training=True)


        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train, X_test, y_train, y_test
    
    
if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    
    df = preprocessor.load_data("C:\\Users\\Himanshu Kumawat\\OneDrive\\Desktop\\CHURN\\notebooks\\telco_comm_churn.csv")
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)    
    
    
    print(f"\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    print(f"Training target: {y_train.shape}")
    print(f"Test target: {y_test.shape}")
    
    # Verify no object columns remain
    print(f"\nFinal Data Types:")
    print(X_train.dtypes.value_counts())
    
    # Sample of first few rows
    print(f"\nSample Data (first 3 rows, first 5 columns):")
    print(X_train.iloc[:3, :5])