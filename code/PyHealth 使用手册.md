# PyHealth 使用手册

----

# 1. PyHealth简介

PyHealth支持MIMIC-III、MIMIC-IV和eICU三类主流医疗数据集。整个Healthcare任务可以分为以下五个阶段：

healthcare tasks in our package follow a **five-stage pipeline**:

> load dataset -> define task function -> build ML/DL model -> model training -> inference

比如一个ML的Pipeline的案例，在MIMIC3数据集上用Transformer：

```
# Task1: Data Load

from pyhealth.datasets import MIMIC3Dataset
mimic3base = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "CONDITIONS"],
    # map all NDC codes to ATC 3-rd level codes in these tables
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)

# Task2: Problem Definition

from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.datasets import split_by_patient, get_dataloader

mimic3sample = mimic3base.set_task(task_fn=drug_recommendation_mimic3_fn) # use default task
train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])

# create dataloaders (torch.data.DataLoader)
train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

# Task3. Model Define
from pyhealth.models import Transformer

model = Transformer(
    dataset=mimic3sample,
    feature_keys=["conditions", "procedures"],
    label_key="drugs",
    mode="multilabel",
)

# Task4. Trainer
from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=50,
    monitor="pr_auc_samples",
)

# Task5. Evaluate
trainer.evaluate(test_loader)
```

一些常见的Baseline Method有：

MLP，CNN，RNN，Transformer，RETAIN，GAMENet， MICRON， SafeDrug， ModelRec， Deepr， STFT+CNN， 1DCNN，TCN，AdaCare，ConCare，StageNet，DrAgent，GRASP.

# 2. Data

在PyHealth里面，数据类型分为三类：Event，Visit和Patient。

## 2.1 Event

event的example：

```
>>> from pyhealth.data import Event
>>> event = Event(
...     code="00069153041",
...     table="PRESCRIPTIONS",
...     vocabulary="NDC",
...     visit_id="v001",
...     patient_id="p001",
...     dosage="250mg",
... )
>>> event
Event with NDC code 00069153041 from table PRESCRIPTIONS
>>> event.attr_dict
{'dosage': '250mg'}
```

## 2.2 Visit

一次Visit有很多的Event

```
from pyhealth.data import Event, Visit
event = Event(
    code="00069153041",
    table="PRESCRIPTIONS",
    vocabulary="NDC",
    visit_id="v001",
    patient_id="p001",
    dosage="250mg",
)
visit = Visit(
    visit_id="v001",
    patient_id="p001",
)
visit.add_event(event)
>>> Visit v001 from patient p001 with 1 events from tables ['PRESCRIPTIONS']
```

## 2.3 Patient

```
>>> from pyhealth.data import Event, Visit, Patient
>>> event = Event(
...     code="00069153041",
...     table="PRESCRIPTIONS",
...     vocabulary="NDC",
...     visit_id="v001",
...     patient_id="p001",
...     dosage="250mg",
... )
>>> visit = Visit(
...     visit_id="v001",
...     patient_id="p001",
... )
>>> visit.add_event(event)
>>> patient = Patient(
...     patient_id="p001",
... )
>>> patient.add_visit(visit)
>>> patient
Patient p001 with 1 visits
```

# 3. Task

1. Drug Recommendation [Yang et al. IJCAI 2021a, Yang et al. IJCAI 2021b, Shang et al. AAAI 2020]
2. Readmission Prediction [Choi et al. AAAI 2021]
3. Mortality Prediction [Choi et al. AAAI 2021]
4. Length of Stay Prediction



PyHealth提供的任务都是基于单次visit（visit level而非patient level）的预测，而不是基于历史visit信息的预测；



改进：基于历史信息来预测下一个时刻的情况 （修改PyHealth代码接口）

```
def patient_level_mortality_prediction_mimic4(patient, dataset='mimic4'):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # if the patient only has one visit, we drop it    
    if len(patient) == 1:
        return []

    # step 1: define label
    idx_last_visit = len(patient) - 1
    if patient[idx_last_visit].discharge_status not in [0, 1]:
        mortality_label = 0
    else:
        mortality_label = int(patient[idx_last_visit].discharge_status)

    # step 2: obtain features
    conditions_merged = []
    procedures_merged = []
    drugs_merged = []
    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        if dataset == 'mimic3':
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
        if dataset == 'mimic4':
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            drugs = visit.get_code_list(table="prescriptions")

        conditions_merged += [conditions]
        procedures_merged += [procedures]
        drugs_merged += [drugs]

    if drugs_merged == [] or procedures_merged == [] or conditions_merged == []:        # todo
        return []

    uniq_conditions = conditions_merged
    uniq_procedures = procedures_merged
    uniq_drugs = drugs_merged

    # step 3: exclusion criteria
    if len(uniq_conditions) * len(uniq_procedures) * len(uniq_drugs) == 0:
        return []

    # step 4: assemble the sample
    samples.append(
        {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": uniq_conditions,
            "procedures": uniq_procedures,
            "drugs": uniq_drugs,
            "label": mortality_label,
        }
    )
    return samples

```



# 4. GRACE

1. x => recon_x:  Decoder
2. GAN:  sample_x => decoder 
3. Real + Fake x => RNN  Encoder => z => 自监督



real x => RNN  Encoder 



![image-20230714113632079](/Users/thinkerjiang/Library/Application Support/typora-user-images/image-20230714113632079.png)

实现情况：将GRACE的数据接入方式修改为PyHealth的统一接口，然后复现过程中将Transformer换成了MLP。

注意事项：不同步数拥有不同时间步长，所以需要对数据做Padding操作以统一计算



实验结果：按照8:1:1的train-valid-test的划分，在MIMIC III上进行实验，实验结果如下：

RNN：pr_auc: 0.1814   roc_auc: 0.6148   **f1: 0.2178**

RNN+GRACE：**pr_auc: 0.1861  roc_auc: 0.6421**   f1: 0.1594

![image-20230714113359459](/Users/thinkerjiang/Library/Application Support/typora-user-images/image-20230714113359459.png)



实验结果：按照6:2:2的train-valid-test的划分，在MIMIC III上进行实验，实验结果如下：

RNN：pr_auc: 0.1291   **roc_auc: 0.6389**   f1: 0.0964

RNN+GRACE：**pr_auc: 0.1763**  roc_auc: 0.6309   **f1: 0.1676**



实验结果：按照4:3:3的train-valid-test的划分，在MIMIC III上进行实验，实验结果如下：

RNN：pr_auc: 0.1291   **roc_auc: 0.6389**   f1: 0.0964

RNN+GRACE：**pr_auc: 0.1763**  roc_auc: 0.6309   **f1: 0.1676**







其他：GRACE停止生长的条件：Then, we stop the generating process when the time interval between the two successive visits is less than the threshold, which is defined as the smallest time interval in real visit sequences



