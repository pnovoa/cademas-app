from cademas.io.load_models import load_adm_metadata

if __name__ == "__main__":
    adms = load_adm_metadata("../configs/example_models.json")

    print(f"Loaded {len(adms)} ADMs\n")

    for adm in adms:
        print("ADM ID:", adm.adm_id)
        print("Department:", adm.department)
        print("Model type:", adm.model_type)
        print("Target class:", adm.target_class)
        print("Features:", adm.features)
        print("AUC:", adm.performance.get("AUC"))
        print("-" * 40)