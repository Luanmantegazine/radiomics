import pandas as pd

old_path = "/Users/luanr/vscode/radiomics/supervised_dataset/supervised_radiomics_sessions.csv"
new_path = "/Users/luanr/vscode/radiomics/supervised_final/supervised_radiomics_sessions.csv"

output_path = "/Users/luanr/vscode/radiomics/supervised_radiomics_sessions_final.csv"

old = pd.read_csv(old_path, low_memory=False)
new = pd.read_csv(new_path, low_memory=False)

print("Antigo:", old.shape)
print("Novo:", new.shape)

# Garantir que os schemas são iguais
assert list(old.columns) == list(new.columns), "Os CSVs possuem colunas diferentes."

final = pd.concat([old, new], ignore_index=True)

# Garantir que nenhuma sessão foi duplicada
duplicated = final["session_id"].duplicated(keep=False)

if duplicated.any():
    print("Sessões duplicadas encontradas:")
    print(
        final.loc[
            duplicated,
            ["subject_id", "session_id", "supervised_label"]
        ].sort_values("session_id")
    )
    raise ValueError("Existem session_id duplicados.")

final.to_csv(output_path, index=False)

print("\nDataset consolidado salvo em:")
print(output_path)

print("\nShape final:", final.shape)
print("Sessões:", final["session_id"].nunique())
print("Sujeitos:", final["subject_id"].nunique())

print("\nDistribuição clínica:")
print(final["supervised_label"].value_counts(dropna=False))

print("\nElegíveis para treino:")
eligible = final[final["training_eligible"] == True]
print(eligible["supervised_label"].value_counts(dropna=False))
print("Total elegível:", len(eligible))