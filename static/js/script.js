
  const FIELDS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", 
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings", 
    "Collateral_Value", "Loan_Amount", "Loan_Term", "Education_Level",
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Gender", "Employer_Category"
  ];

  async function checkLoan() {
    const btn = document.getElementById("predictBtn");
    const errBox = document.getElementById("errorMsg");
    const resDiv = document.getElementById("result");
    
    errBox.style.display = "none";
    resDiv.style.display = "none";

    const data = {};
    for (const f of FIELDS) {
      const val = document.getElementById(f).value;
      if (val === "") {
        errBox.textContent = `Please fill in: ${f.replace(/_/g, ' ')}`;
        errBox.style.display = "block";
        return;
      }
      data[f] = val;
    }

    btn.disabled = true;
    btn.textContent = "Processing...";

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      
      const json = await response.json();
      btn.disabled = false;
      btn.textContent = "Analyze Application";

      if (json.error) {
        errBox.textContent = json.error;
        errBox.style.display = "block";
        return;
      }

      resDiv.className = json.approved ? "approved" : "rejected";
      resDiv.style.display = "block";
      document.getElementById("resultTitle").textContent = json.approved ? "✅ Approved" : "❌ Rejected";
      document.getElementById("resultSub").textContent = json.approved ? "Applicant is eligible for the loan." : "Applicant does not meet the requirements.";
      document.getElementById("barFill").style.width = json.confidence + "%";
      document.getElementById("barLabel").textContent = `Confidence: ${json.confidence}%`;

    } catch (e) {
      btn.disabled = false;
      btn.textContent = "Analyze Application";
      errBox.textContent = "Server Connection Failed. Is your Flask app running?";
      errBox.style.display = "block";
    }
  }
