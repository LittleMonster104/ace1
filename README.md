1. Dataset Overview
Item    Description
Number of high-quality images    160,490
Number of image-text pairs    481,470
Descriptions per image    3
Description perspectives    Instructional context / Knowledge content / Visual presentation
Primary language    Chinese; English translations retained in extended fields
Data format    Karpathy-style COCO/Flickr JSON
Fixed splits    Train 144,441 / Val 8,022 / Test 8,027
The dataset is designed to evaluate the cross-domain generalization of vision-language models in non-natural image settings, including classroom presentations, structured pages, and educational knowledge representations. The full dataset contains training, validation, and test splits; the subset used by a specific paper or project should follow the experimental settings reported in the corresponding paper and released code.
2. Data Sources and Coverage
The source data consist of classroom presentation files independently collected and organized by the research group. Each presentation was converted into page-level images, while intermediate metadata such as subject, source file, and page number were retained for traceability. The dataset is not a subset of public image-text retrieval datasets such as COCO or Flickr30K, nor was it obtained by crawling images from public webpages.
The source materials cover a wide range of education-related subjects and instructional settings, including biology, comprehensive secondary education, primary education, chemistry, English, Chinese language and literature, science education, preschool education, mathematics, fine arts, psychology, educational technology, music, history, ideological and political education, humanities education, calligraphy, physical education, physics, special education, and teaching Chinese as a foreign language.
2.1 Subject Distribution
Subject    Images    Subject    Images
Biology    37,093    Comprehensive Secondary Education    25,100
Primary Education    23,623    Chemistry    17,054
English    8,690    Chinese Language and Literature    7,622
Science Education    5,506    Preschool Education    5,483
Mathematics    3,815    Fine Arts    3,556
Psychology    3,029    Educational Technology    2,976
Music    2,589    History    2,407
Ideological and Political Education    2,264    Humanities Education    2,157
Other    1,789    Calligraphy    1,505
Physical Education    1,423    Physics    1,315
Special Education    786    Teaching Chinese as a Foreign Language    708
3. Dataset Construction Pipeline
1.Slide image extraction. Classroom presentations were converted into page-level images, with each page treated as one candidate image and organized by subject directory.
2.Rule-based filtering. Cover slides, ending slides, blank or near-blank pages, adjacent duplicate pages, and pages with limited visual and textual information were removed. Pages with little text were retained when they contained meaningful charts, illustrations, or experimental images.
3.Multimodal semantic annotation. A locally deployed Qwen3.6-27B multimodal model was used to generate Chinese descriptions together with a corresponding English translation for each description.
4.High-quality filtering. Records with failed annotations, incomplete descriptions, quality flags, or uncertain content were removed. Only samples with complete descriptions from all three perspectives and non-empty Chinese and English fields were retained.
5.Format conversion. The final dataset was converted into Karpathy-style JSON compatible with common COCO/Flickr retrieval pipelines, while retaining extended fields related to educational content.
4. Description Design
Each high-quality image is associated with three complementary descriptions. Rather than being simple paraphrases, the descriptions characterize the same instructional slide from different semantic perspectives.
Field / Perspective    Description Objective    Typical Content
subject    Summarize the subject, classroom setting, or instructional topic    Course type, subject area, instructional activity
knowledge    Describe specific knowledge points, learning objectives, or question content    Concepts, formulas, texts, experimental phenomena, questions
teaching_visual    Describe the page layout and visible instructional elements    Charts, illustrations, board writing, experimental images, page structure
The primary annotation field is the Chinese description (raw), while the English translation is stored in the extended field raw_en. If an existing retrieval method reads English captions by default, the raw_en field can be used or an English-version Karpathy JSON file can be exported.
5. Quality Control and Filtering Statistics
After annotation of all candidate samples, the data underwent automatic quality checks and rule-based filtering. The final high-quality version requires each image to contain subject, knowledge, and teaching_visual descriptions, with both Chinese and English content present.
Statistic    Count
Raw candidate records    186,272
Removed due to annotation failure    5,082
Removed due to quality flags    19,877
Removed due to incomplete descriptions    823
Final high-quality images    160,490
Final image-text pairs    481,470
Quality flags include low_information, mostly_text, uncertain_ocr, uncertain_knowledge, and unclear_image. All samples carrying any quality flag were removed from the final high-quality set.
6. Dataset Splits
Using a fixed random seed of 42, the dataset was divided into training, validation, and test sets at a ratio of 90% / 5% / 5%.
Split    Images    Image-text pairs
Train    144,441    433,323
Validation    8,022    24,066
Test    8,027    24,081
To ensure comparability, researchers are encouraged to use the fixed splits and clearly report in their papers or code which images and descriptions are read during the adaptation and evaluation stages.
7. Dataset Directory and Main Files
edu_ppt_hq_dataset/
├── images/
├── annotations/
│   ├── dataset_edu_ppt_hq_karpathy.json
│   ├── dataset_edu_ppt_hq_karpathy.summary.json
│   ├── images.jsonl
│   ├── pairs.jsonl
│   └── summary.json
├── metadata/
└── README.md

The main annotation file is annotations/dataset_edu_ppt_hq_karpathy.json. Image paths follow images/{filepath}/{filename}, for example, images/comprehensive_secondary_education/comprehensive_secondary_education_00001.jpg.
7.1 File Sizes
File or directory    Approximate size
Full dataset directory    Approx. 27 GB
images/    Approx. 27 GB
dataset_edu_ppt_hq_karpathy.json    Approx. 257 MB
images.jsonl    Approx. 313 MB
pairs.jsonl    Approx. 188 MB
8. Karpathy Annotation Format
The main annotation file is structurally compatible with the commonly used dataset_coco.json and dataset_flickr30k.json files. Standard retrieval code typically reads filename, filepath, imgid, sentences, sentids, and split; the additional fields do not affect implementations that read only the standard fields.
Level    Field    Description
Top level    dataset    Dataset name: edu_ppt_hq
Top level    images    Image sample list
Image    filename / filepath    File name and relative directory
Image    imgid / sentids / split    Image ID, description IDs, and split
Image    edu_path / subject    Original relative path and subject directory
Description    raw / tokens / sentid    Chinese description, tokenization results, and description ID
Description    aspect / raw_en    Description perspective and English translation
8.1 Simplified Example
{
  "filename": "comprehensive_secondary_education_00001.jpg",
  "filepath": "comprehensive_secondary_education",
  "imgid": 0,
  "split": "train",
  "sentences": [
    {"aspect": "subject", "raw": "...", "raw_en": "..."},
    {"aspect": "knowledge", "raw": "...", "raw_en": "..."},
    {"aspect": "teaching_visual", "raw": "...", "raw_en": "..."}
  ]
}

9. Applicable Tasks and Usage Recommendations
Image-text retrieval and semantic alignment evaluation in educational domains;
Zero-shot or cross-domain generalization evaluation of vision-language models on structured pages and specialized knowledge scenarios;
Research on multimodal representation learning, document image understanding, and educational content understanding;
Stratified analysis by subject, knowledge content, or visual presentation type.
Users should clearly specify the language field actually read (raw or raw_en), the number of descriptions used per image, the definition of positive image-text matches, and whether the adaptation and evaluation sets overlap. Because these choices may substantially affect retrieval metrics, they should be fully disclosed in experimental reports.
10. Limitations and Data Management
The descriptions are generated automatically by a multimodal model. Despite rule-based filtering, semantic bias, factual errors, or translation inaccuracies may remain.
The subject distribution is imbalanced, and the relationships among text, charts, and visual elements in presentation slides are complex. Aggregate results may therefore be influenced by high-frequency categories and the characteristics of structured pages.
External sharing, access permissions, copyright, and privacy handling should follow the actual agreements between the research group and the material providers. In addition, accidental overlap cannot be conclusively excluded because the pretraining corpora of the evaluated models are not publicly available.
