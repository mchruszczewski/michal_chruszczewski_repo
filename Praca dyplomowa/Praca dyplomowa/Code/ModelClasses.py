import pandas as pd
import numpy as np
import os
import openpyxl
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report, precision_recall_curve, f1_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


import pandas as pd

class DataProcessing:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def read_data(self):
        try:
            excel = pd.ExcelFile(self.data_path)
            data_frames = [pd.read_excel(excel, sheet_name=sheet) for sheet in excel.sheet_names]
            self.data = pd.concat(data_frames, ignore_index=True)
            return self.data
        except FileNotFoundError:
            print(f"file {self.data_path} not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def prepare_data(self):
        if self.data is None:
            print("Data not loaded. Please use read_data() first.")
            return None

        try:
            self.data['label'] = 'relegation'
            self.data.loc[self.data['Rk'] < 16, 'label'] = 'league_average'
            self.data.loc[self.data['Rk'] < 10, 'label'] = 'european_cups'
            self.data.loc[self.data['Rk'] < 5, 'label'] = 'champions_group'

            exclude_substrings = ['Pts', 'W', 'L', 'D', 'MP', 'Squad', 'Top Team Scorer', 'Notes', 'Goalkeeper', '90s',
                                'Min%', 'Mn/Start','GD','Rk', 'Pts/MP', 'xGD', 'xGD/90']
            non_basic_stats = [col for col in self.data.columns if (col not in exclude_substrings)]
            self.data= self.data[non_basic_stats]

            categories=[]
            for column in self.data.columns:
                if self.data[column].nunique() <11:
                    categories.append(column)

            self.data[categories]= self.data[categories].astype('category')
            non_categories = self.data.select_dtypes(exclude=['category']).columns
            self.data[non_categories]= self.data[non_categories].apply(lambda col: col.fillna(col.mean()), axis=0)

            cols_percent= [col for col in self.data.columns if '%' in col and col not in categories]
            self.data[cols_percent]= self.data[cols_percent]/100 
            return self.data
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def model_data (self):

        try:

            X = self.data.drop('label', axis=1)
            y = self.data['label']
            return [X,y]
        
        except Exception as e:
            print(f"Error: {e}")
            return None
        
class ModelClass():

    label_encoder = LabelEncoder()
    def __init__(self, X, y):
        self.X= X
        self.y= y
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        self.numerical_cols = [col for col in self.X.columns if self.X[col].dtype != 'category']
        self.cat_cols = [col for col in self.X.columns if self.X[col].dtype == 'category']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ])
        self.class_names= self.label_encoder.classes_


    def glm_model (self):
        model_glm = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
        pipeline_glm = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model_glm)
                    ])
        
        y_proba_glm = cross_val_predict(pipeline_glm, self.X, self.y_encoded, cv=5, method='predict_proba')
        y_bin_glm = label_binarize(self.y_encoded, classes=np.unique(self.y_encoded))
        n_classes = y_bin_glm.shape[1]
        

        fpr_glm, tpr_glm, roc_auc_glm, thresholds_glm = {}, {}, {}, {}
        best_thresholds_glm = {}

        for i in range(n_classes):
            fpr_glm[i], tpr_glm[i], thresholds_glm[i] = roc_curve(y_bin_glm[:, i], y_proba_glm[:, i])
            roc_auc_glm[i] = auc(fpr_glm[i], tpr_glm[i])
            j_scores = tpr_glm[i] - fpr_glm[i]
            j_best_index = np.argmax(j_scores)
            best_thresholds_glm[i] = thresholds_glm[i][j_best_index]
        all_fpr_glm = np.unique(np.concatenate([fpr_glm[i] for i in range(n_classes)]))
        mean_tpr_glm = np.zeros_like(all_fpr_glm)

        for i in range(n_classes):
            mean_tpr_glm += np.interp(all_fpr_glm, fpr_glm[i], tpr_glm[i])

        mean_tpr_glm /= n_classes
        roc_glm = all_fpr_glm
        auc_glm = auc(roc_glm, mean_tpr_glm)

        y_pred_custom_glm = np.zeros_like(self.y_encoded)
        for i in range(n_classes):
            y_pred_custom_glm[y_proba_glm[:, i] >= best_thresholds_glm[i]] = i

        return {'y_encoded':self.y_encoded, 
                'class_names':self.class_names,
                'y_pred_custom_glm': y_pred_custom_glm,
                'n_classes': n_classes,
                'fpr_glm':fpr_glm,
                'tpr_glm':tpr_glm,
                'best_thresholds_glm':best_thresholds_glm,
                'roc_glm':roc_glm,
                'mean_tpr_glm':mean_tpr_glm,
                'auc_glm': auc_glm,
                'roc_auc_glm':roc_auc_glm}
    
    def decision_tree_model(self):

        model_tree = DecisionTreeClassifier(random_state=42)
        pipeline_tree = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model_tree)
        ])
        
        y_proba_tree = cross_val_predict(pipeline_tree, self.X, self.y_encoded, cv=5, method='predict_proba')
        y_bin_tree = label_binarize(self.y_encoded, classes=np.unique(self.y_encoded))
        n_classes = y_bin_tree.shape[1]

        fpr_tree, tpr_tree, roc_auc_tree, thresholds_tree = {}, {}, {}, {}
        best_thresholds_tree = {}

        for i in range(n_classes):
            fpr_tree[i], tpr_tree[i], thresholds_tree[i] = roc_curve(y_bin_tree[:, i], y_proba_tree[:, i])
            roc_auc_tree[i] = auc(fpr_tree[i], tpr_tree[i])
            j_scores = tpr_tree[i] - fpr_tree[i]
            j_best_index = np.argmax(j_scores)
            best_thresholds_tree[i] = thresholds_tree[i][j_best_index]
        
        all_fpr_tree = np.unique(np.concatenate([fpr_tree[i] for i in range(n_classes)]))
        mean_tpr_tree = np.zeros_like(all_fpr_tree)

        for i in range(n_classes):
            mean_tpr_tree += np.interp(all_fpr_tree, fpr_tree[i], tpr_tree[i])

        mean_tpr_tree /= n_classes
        roc_tree = all_fpr_tree
        auc_tree = auc(roc_tree, mean_tpr_tree)

        y_pred_custom_tree = np.zeros_like(self.y_encoded)
        for i in range(n_classes):
            y_pred_custom_tree[y_proba_tree[:, i] >= best_thresholds_tree[i]] = i

        return {
            'y_encoded': self.y_encoded,
            'class_names': self.class_names,
            'y_pred_custom_tree': y_pred_custom_tree,
            'n_classes': n_classes,
            'fpr_tree': fpr_tree,
            'tpr_tree': tpr_tree,
            'best_thresholds_tree': best_thresholds_tree,
            'roc_tree': roc_tree,
            'mean_tpr_tree': mean_tpr_tree,
            'auc_tree': auc_tree,
            'roc_auc_tree': roc_auc_tree,
            'model_tree':model_tree
        }
    
    
    def random_forest_model(self):
        model_rf = RandomForestClassifier(random_state=42, n_estimators=1000)
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model_rf)
        ])
        
        y_proba_rf = cross_val_predict(pipeline_rf, self.X, self.y_encoded, cv=5, method='predict_proba')
        y_bin_rf = label_binarize(self.y_encoded, classes=np.unique(self.y_encoded))
        n_classes = y_bin_rf.shape[1]

        fpr_rf, tpr_rf, roc_auc_rf, thresholds_rf = {}, {}, {}, {}
        best_thresholds_rf = {}

        for i in range(n_classes):
            fpr_rf[i], tpr_rf[i], thresholds_rf[i] = roc_curve(y_bin_rf[:, i], y_proba_rf[:, i])
            roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
            j_scores = tpr_rf[i] - fpr_rf[i]
            j_best_index = np.argmax(j_scores)
            best_thresholds_rf[i] = thresholds_rf[i][j_best_index]
        
        all_fpr_rf = np.unique(np.concatenate([fpr_rf[i] for i in range(n_classes)]))
        mean_tpr_rf = np.zeros_like(all_fpr_rf)

        for i in range(n_classes):
            mean_tpr_rf += np.interp(all_fpr_rf, fpr_rf[i], tpr_rf[i])

        mean_tpr_rf /= n_classes
        roc_rf = all_fpr_rf
        auc_rf = auc(roc_rf, mean_tpr_rf)

        y_pred_custom_rf = np.zeros_like(self.y_encoded)
        for i in range(n_classes):
            y_pred_custom_rf[y_proba_rf[:, i] >= best_thresholds_rf[i]] = i
        
        self.preprocessor.fit(self.X)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        feature_importances = np.zeros(len(self.preprocessor.get_feature_names_out()))

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y_encoded[train_index], self.y_encoded[test_index]
            
            X_train_transformed = self.preprocessor.transform(X_train)
            model_rf.fit(X_train_transformed, y_train)
            
            feature_importances += model_rf.feature_importances_
        
        feature_importances /= kf.get_n_splits()

        # Tworzenie DataFrame z wynikami
        feature_importance_df = pd.DataFrame({
            'feature': self.preprocessor.get_feature_names_out(),
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)

        return {
            'y_encoded': self.y_encoded,
            'class_names': self.class_names,
            'y_pred_custom_rf': y_pred_custom_rf,
            'n_classes': n_classes,
            'fpr_rf': fpr_rf,
            'tpr_rf': tpr_rf,
            'best_thresholds_rf': best_thresholds_rf,
            'roc_rf': roc_rf,
            'mean_tpr_rf': mean_tpr_rf,
            'auc_rf': auc_rf,
            'roc_auc_rf': roc_auc_rf,
            'feature_importance_df': feature_importance_df  # Nowy klucz w słowniku
        }









      

