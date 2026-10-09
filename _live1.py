from dental.case_gen import generate_case
b = generate_case(difficulty="mid")
p = b.patient
print("PATIENT:", p.age, p.sex, "|", p.complaint)
print("medical:", p.medical_history)
print("drugs:", p.drug_history, "| allergies:", p.allergies)
print("hidden:", p.hidden_findings)
print("diagnosis:", p.intended_diagnosis)
print("differentials:", p.differentials)
print("investigations:", p.investigation_results)
print("rubric phases:", [(r.phase, r.max_score) for r in b.rubric])
print("history criteria:", [r.criteria for r in b.rubric if r.phase=="history"])
