class FeatureEngineering:
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        job_roles_features = [col for col in df.columns if "Job" in col]
        for feature in ['Tenure', 'Salary', "Average Monthly Hours Worked"]:
            for role in job_roles_features:
                df[role + "_" + feature] = df[role] * df[feature]

        martial_status = [col for col in df.columns if "Marital Status" in col]
        work_life_balance = [col for col in df.columns if "Work-Life Balance" in col]
        educational_level = [col for col in df.columns if "Education Level" in col]
        for features in [martial_status, work_life_balance, educational_level]:
            for feature in features:
                df["Average Monthly Hours Worked_" + feature] = df['Average Monthly Hours Worked'] * df[feature]

        df['Average Monthly Hours Worked and Distance from Home'] = df['Average Monthly Hours Worked'] + df[
            'Distance from Home']
        return df
