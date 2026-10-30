"""Generate a native Power BI project using the documented PBIR/TMSL formats."""
import json
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "powerbi"
R = P / "ChurnRisk.Report"
S = P / "ChurnRisk.SemanticModel"
BASE = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/"

def identifier(name):
    return hashlib.sha1(name.encode()).hexdigest()[:20]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def lit(value):
    v = "true" if value is True else "false" if value is False else f"'{value}'" if isinstance(value, str) else str(value) + "D"
    return {"expr": {"Literal": {"Value": v}}}


def color(value):
    return {"solid": {"color": lit(value)}}


def field(name, measure=False):
    return {"Measure" if measure else "Column": {"Expression": {"SourceRef": {"Entity": "Customers"}}, "Property": name}}


def projection(name, measure=False):
    return {"field": field(name, measure), "queryRef": f"Customers.{name}", "nativeQueryRef": name}


def main():
    save(P / "ChurnRisk.pbip", {"version": "1.0", "artifacts": [{"report": {"path": "ChurnRisk.Report"}}], "settings": {"enableAutoRecovery": True}})
    save(R / "definition.pbir", {"version": "4.0", "datasetReference": {"byPath": {"path": "../ChurnRisk.SemanticModel"}}})
    save(S / "definition.pbism", {"version": "1.0", "settings": {"qnaEnabled": False}})
    names = {"customerID": "string", "Contract": "string", "PaymentMethod": "string", "InternetService": "string", "tenure": "int64",
             "MonthlyCharges": "double", "RiskScore": "double", "RiskBand": "string", "RiskWeightedMonthlyCharges": "double",
             "TenureBand": "string", "ModelCohort": "string", "ScoredAtUTC": "string", "SourceMode": "string", "ModelVersion": "string"}
    columns = [{"name": k, "dataType": v, "sourceColumn": k, "summarizeBy": "none",
                **({"formatString": "0.0%"} if k == "RiskScore" else {"formatString": "#,0.00"} if v == "double" else {})} for k, v in names.items()]
    measures = {
        "Active Customers": ("COUNTROWS(Customers)", "#,0"),
        "Average Risk": ("AVERAGE(Customers[RiskScore])", "0.0%"),
        "High Risk Customers": ('CALCULATE(COUNTROWS(Customers), KEEPFILTERS(Customers[RiskScore] >= 0.5))', "#,0"),
        "Monthly Charges CU": ("SUM(Customers[MonthlyCharges])", "#,0.00"),
        "Revenue at Risk Proxy CU": ("SUMX(Customers, Customers[MonthlyCharges] * Customers[RiskScore])", "#,0.00"),
        "High Risk Monthly Charges CU": ("CALCULATE(SUM(Customers[MonthlyCharges]), KEEPFILTERS(Customers[RiskScore] >= 0.5))", "#,0.00"),
        "Last Scored UTC": ("MAX(Customers[ScoredAtUTC])", None)}
    mtypes = {"string": "type text", "int64": "Int64.Type", "double": "type number"}
    types = ", ".join('{"'+k+'", '+mtypes[v]+'}' for k, v in names.items())
    expression = ['let', '    Source = Csv.Document(File.Contents(DataPath), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
                  '    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
                  '    Typed = Table.TransformColumnTypes(Headers, {'+types+'}, "en-US")', 'in', '    Typed']
    save(S / "model.bim", {"name": "ChurnRisk", "compatibilityLevel": 1600, "model": {
        "culture": "en-US", "defaultPowerBIDataSourceVersion": "powerBI_V3", "sourceQueryCulture": "en-US",
        "expressions": [{"name": "DataPath", "kind": "m", "expression": '"'+str(P / 'data/customer_risk.csv')+'" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'}],
        "tables": [{"name": "Customers", "columns": columns,
                    "measures": [{"name": k, "expression": v[0], **({"formatString": v[1]} if v[1] else {})} for k, v in measures.items()],
                    "partitions": [{"name": "Customers", "mode": "import", "source": {"type": "m", "expression": expression}}]}]}})
    save(R / "definition/version.json", {"$schema": BASE+"versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
    # Keep the base theme registered by the installed Power BI Desktop version.
    report_template = P / "report_template.json"
    save(R / "definition/report.json", json.loads(report_template.read_text(encoding="utf-8-sig")))
    pages = ["overview", "customers", "definitions"]
    save(R / "definition/pages/pages.json", {"$schema": BASE+"pagesMetadata/1.1.0/schema.json", "pageOrder": [identifier(p) for p in pages], "activePageName": identifier("overview")})
    titles = ["01  Risk overview", "02  Customer explorer", "03  Definitions & freshness"]
    for page, title in zip(pages, titles):
        save(R / f"definition/pages/{identifier(page)}/page.json", {"$schema": BASE+"page/2.0.0/schema.json", "name": identifier(page),
             "displayName": title, "displayOption": "FitToPage", "width": 1280, "height": 900,
             "objects": {"background": [{"properties": {"color": color("#F1F5F9"), "transparency": lit(0)}}]}})
    def visual(page, name, kind, x,y,w,h, title, roles=None, objects=None, sort=None):
        objects = dict(objects or {})
        if kind == "card":
            objects["categoryLabels"] = [{"properties": {"show": lit(False)}}]
        if kind == "slicer":
            objects["header"] = [{"properties": {"show": lit(False)}}]
        if kind == "tableEx":
            objects["grid"] = [{"properties": {"textSize": lit(12)}}]
        v = {"visualType": kind, "drillFilterOtherVisuals": True,
             "visualContainerObjects": {"title": [{"properties": {"show": lit(True), "text": lit(title), "fontSize": lit(12), "fontColor": color("#15364A")}}],
                                         "background": [{"properties": {"show": lit(True), "color": color("#FFFFFF"), "transparency": lit(0)}}]}}
        if roles:
            v["query"] = {"queryState": {role: {"projections": [projection(n,m) for n,m in fields]} for role,fields in roles.items()}}
            if sort:
                v["query"]["sortDefinition"] = {"sort": [{"field": field(*sort), "direction": "Descending"}]}
        if objects: v["objects"] = objects
        save(R / f"definition/pages/{identifier(page)}/visuals/{identifier(name)}/visual.json", {"$schema": BASE+"visualContainer/2.4.0/schema.json", "name": identifier(name),
             "position": {"x": x, "y": y, "width": w, "height": h, "z": 0, "tabOrder": 0}, "visual": v})
    def text(page, name, text, x,y,w,h, size=14):
        visual(page,name,"textbox",x,y,w,h,"",objects={"general":[{"properties":{"paragraphs":[{"textRuns":[{"value":text,"textStyle":{"fontSize":f"{size}pt","fontFamily":"Segoe UI","color":"#15364A"}}]}]}}]})
    text("overview","heading","TELECOM  /  CHURN RISK & REVENUE EXPOSURE",20,12,1240,50,23)
    text("overview","notice","HISTORICAL DEMO  •  Active-at-snapshot customers  •  Scores are uncalibrated  •  CU = source currency units",20,64,1240,40,11)
    for i,(col,title) in enumerate([("Contract","Contract"),("RiskBand","Risk band"),("InternetService","Internet service"),("ModelCohort","Model cohort")]):
        visual("overview",f"slice{i}","slicer",20+i*310,110,300,95,title,{"Values":[(col,False)]}, {"data":[{"properties":{"mode":lit("Dropdown")}}]})
    for i,(measure,title) in enumerate([("Active Customers","ACTIVE CUSTOMERS"),("High Risk Customers","HIGH RISK  |  SCORE ≥ 0.50"),("Revenue at Risk Proxy CU","SCORE-WEIGHTED EXPOSURE · CU"),("High Risk Monthly Charges CU","HIGH-RISK MONTHLY CHARGES · CU")]):
        visual("overview",f"kpi{i}","card",20+i*310,220,300,120,title,{"Values":[(measure,True)]}, {"labels":[{"properties":{"fontSize":lit(27),"color":color("#146B7A"),"labelDisplayUnits":lit(1)}}]})
    for i,(col,title) in enumerate([("Contract","Weighted monthly exposure by contract"),("RiskBand","Customers by risk band")]):
        measure = "Revenue at Risk Proxy CU" if i==0 else "Active Customers"
        visual("overview",f"bar{i}","clusteredBarChart",20+i*620,355,610,250,title,{"Category":[(col,False)],"Y":[(measure,True)]},sort=(measure,True))
    visual("overview","risk_table","tableEx",20,620,1240,230,"CUSTOMER PRIORITIES  |  Click a segment above to filter",{"Values":[("customerID",False),("Contract",False),("RiskBand",False),("RiskScore",False),("MonthlyCharges",False),("RiskWeightedMonthlyCharges",False),("ModelCohort",False)]},sort=("RiskScore",False))
    text("overview","footer","Refresh: rerun score_dashboard.py, then Home > Refresh. No live feed connected. Revenue exposure is a proxy, not forecast loss.",20,855,1240,38,10)
    text("customers","heading2","CUSTOMER EXPLORER  /  FILTER • SORT • REVIEW",20,12,1240,55,23)
    for i,col in enumerate(["RiskBand","Contract","PaymentMethod","TenureBand"]):
        visual("customers",f"filter{i}","slicer",20+i*310,80,300,100,col,{"Values":[(col,False)]},{"data":[{"properties":{"mode":lit("Dropdown")}}]})
    visual("customers","detail","tableEx",20,200,1240,620,"Active-at-snapshot records; training and evaluation cohorts are identified",{"Values":[(c,False) for c in ["customerID","RiskScore","RiskBand","MonthlyCharges","RiskWeightedMonthlyCharges","Contract","PaymentMethod","InternetService","tenure","ModelCohort"]]},sort=("RiskScore",False))
    text("customers","footer2","High ≥ 0.50 • Medium 0.30–0.50 • Low < 0.30. Illustrative thresholds, not a validated outreach policy. Ctrl+click supports multiple selections.",20,835,1240,55,11)
    text("definitions","heading3","DEFINITIONS  /  WHAT THESE NUMBERS MEAN",20,15,1240,60,23)
    paragraphs = [
      "SCOPE: Historical Kaggle snapshot. Only Churn = No records are shown as active at the snapshot. Historical churners are excluded from current-style exposure totals.",
      "RISK SCORE: Saved Random Forest output, on a 0–1 scale. It is not calibrated and has no validated future prediction horizon. Training-cohort scores are in-sample demonstrations.",
      "REVENUE AT RISK PROXY: Sum of MonthlyCharges × RiskScore. This is a score-weighted monthly charge estimate in source currency units (CU), not expected loss or an annual forecast.",
      "HIGH-RISK MONTHLY CHARGES: Sum of MonthlyCharges for score ≥ 0.50. This is gross charge exposure, not an assertion that all high-risk customers will leave.",
      "FRESHNESS: Scored time is when the script ran; it is NOT when the source customer records were observed. Refresh reads the latest local CSV. No live production feed is connected.",
      "LIVE DEPLOYMENT: Requires current active-customer events, scoring infrastructure, a supported database/DirectQuery source, and Power BI refresh configuration. None is silently simulated here."]
    for i,p in enumerate(paragraphs): text("definitions",f"note{i}",p,30,100+i*100,1220,90,15)
    visual("definitions","timestamp","card",30,720,760,115,"LAST SCORING RUN (UTC)",{"Values":[("Last Scored UTC",True)]})
    print("Generated", P / "ChurnRisk.pbip")


if __name__ == "__main__":
    main()
