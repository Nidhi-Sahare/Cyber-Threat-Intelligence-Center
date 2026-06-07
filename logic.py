import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


COLUMN_NAMES = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'attack',
    'difficulty'
]


class CyberThreatModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2,random_state=42)
        self.kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
    def load_data(self):
        train = pd.read_csv("dataset/KDDTrain+_20Percent.txt",names=COLUMN_NAMES)
        test = pd.read_csv("dataset/KDDTest+.txt",names=COLUMN_NAMES)
        return pd.concat([train, test],ignore_index=True)

    def preprocess(self):
        df = self.load_data()
        df.drop(columns=["difficulty"],inplace=True)
        self.original_df = df.copy()
        categorical = ["protocol_type","service","flag"]
        self.label_encoders = {}
        for col in categorical:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        X = df.drop(columns=["attack"])
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.df = df
        self.X_pca = X_pca
        return X_pca
    def run_kmeans(self):
        clusters = self.kmeans.fit_predict(self.X_pca)
        self.df["cluster"] = clusters
        return clusters
    def get_metrics(self):
        total = len(self.original_df)
        attacks = len(self.original_df[self.original_df["attack"] != "normal"])
        normal = len(self.original_df[self.original_df["attack"] == "normal"])
        return {
            "total_records": total,
            "attack_records": attacks,
            "normal_records": normal,
            "clusters": len(self.df["cluster"].unique())
        }

    def threat_summary(self):
        attacks = self.original_df[self.original_df["attack"] != "normal"]
        top_attack = (attacks["attack"].value_counts().idxmax())
        attack_percent = round(len(attacks) * 100 /len(self.original_df),2)
        return {"top_attack": top_attack,"attack_percent": attack_percent}

    def cluster_summary(self):
        return self.df.groupby("cluster").size()
    def cluster_risk_levels(self):
        risks = {}
        for cluster in self.df["cluster"].unique():
            cluster_rows = self.df[self.df["cluster"] == cluster]
            attack_count = len(cluster_rows[self.original_df.loc[cluster_rows.index,"attack"] != "normal"])
            ratio = (attack_count /len(cluster_rows))
            if ratio > 0.70:
                risks[cluster] = {"label":"HIGH RISK"}
            elif ratio > 0.30:
                risks[cluster] = {"label":"MEDIUM RISK"}
            else:
                risks[cluster] = {"label":"LOW RISK"}
        return risks

    def get_pca_data(self):
        return pd.DataFrame({"PC1": self.X_pca[:,0],"PC2": self.X_pca[:,1],"Cluster": self.df["cluster"]}

    def analyze_cluster(self, cluster_id):
        rows = self.df[self.df["cluster"] == cluster_id]
        attacks = self.original_df.loc[rows.index,"attack"]
        return attacks.value_counts().head(10)

    def generate_rules(self):
        rule_df = self.original_df[["protocol_type","service","flag","attack"]].copy()
        basket = pd.get_dummies(rule_df.astype(str))
        frequent = apriori(basket,min_support=0.05,use_colnames=True)
        rules = association_rules(frequent,metric="confidence",min_threshold=0.6)
        rules = rules.sort_values(by="lift",ascending=False)
        rules = rules[["antecedents","consequents","support","confidence","lift"]]
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.head(20)
