# Create dataset from Citation Network Dataset

We use the Citation Network Dataset v10 from "https://www.aminer.org/citation"

The output dataset contains all interested conferences since 2000 (format: id, conf, year, author, title, n_citation, reference, abstract)

Example output: 

```
[
    {
    "id": "09d09df9-c52b-42dd-8b7c-05c902697027",
    "conf": "north american chapter of the association for computational linguistics",
    "year": 2012,
    "authors": [
        "Carmen Banea",
        "Samer Hassan",
        "Michael Mohler",
        "Rada Mihalcea"
        ],
    "title": "UNT: A Supervised Synergistic Approach to Semantic Text Similarity",
    "n_citation": 50,
    "references": [
        "01b486c4-8955-403b-a0c6-1de74298b215",
        "11d6d688-f542-4d4e-8a9e-50150c7f08e3",
        "1268112c-a3a5-411d-9469-c9937aed533e",
        "1ce7a9a3-91c4-45d6-984a-e1d240fd81aa",
        "1f73723c-b904-4d93-8045-d8de3772fb27",
        "2e5b14fa-c54a-4191-9711-6dda5f0eb75d",
        "4a23bcc9-2320-4889-b7c3-6c01f4204051",
        "540f653d-dc81-4160-9755-3cd96bc46bb0",
        "73f6b15e-509c-4f2f-9160-35cab954ce59",
        "785adb42-a91c-485d-866c-69385a666083",
        "7e4dba22-3e36-462f-9220-a8ccd8b75bd4",
        "8026f56a-a93e-4933-8ead-c9aa9e3f0498",
        "8876115b-93b8-4f61-b5b2-46e9e273b74b",
        "892d5d27-8e30-4ab2-b157-3eb1639e2a1d",
        "aafffa69-1bc3-4d18-8f3b-296c03a13557",
        "adaaafab-aa7a-4b1e-842d-e29c8c2f049b",
        "b23adb78-65ed-4bbb-bce9-fdd70d14699e",
        "b855e4aa-03af-4fda-95a9-269138319fc1",
        "ccd672d4-2f77-471e-abfc-cae8abd6ec16",
        "cd34e1c4-2cdb-42b6-bb14-ee7cea07f58b"
        ],
    "abstract": "This paper presents the systems that we participated with in the Semantic Text Similarity task at SEMEVAL 2012. Based on prior research in semantic similarity and relatedness, we combine various methods in a machine learning framework. The three variations submitted during the task evaluation period ranked number 5, 9 and 14 among the 89 participating systems. Our evaluations show that corpus-based methods display a more robust behavior on the training data, yet combining a variety of methods allows a learning algorithm to achieve a superior decision than that achievable by any of the individual parts."
    },
    ...
]
```
