# Study 06: All Images Selected-Subset Random Forest

## מה זה הסטאדי הזה

Study 06 הוא המשך ישיר של קו המחקר ב-`attempts/second attempt`.

הוא נבנה כדי לענות על שאלה מאוד ממוקדת:

- אחרי ש-Study 04 הראה שאפשר ללמוד מודל depth-only חזק על מדגם מאוזן,
  האם אותה גישה עדיין עובדת גם על **כל הדאטה**, ובו-זמנית עם
  **feature space פשוט יותר**?

במילים אחרות:

- Study 04 בדק capability על benchmark מסודר ומאוזן
- Study 06 בודק robustness ו-practicality על כל התמונות

## איך זה ממשיך את מה שעשינו קודם

השרשרת עד לכאן הייתה:

1. Study 01 + Study 02:
   - בדקנו איך נכון לייצג את ה-relative depth של הרחפן.
   - התמקדנו בהשוואת zoom contexts, aggregation methods, ו-score fields.

2. Study 03:
   - בדקנו האם שילוב multiscale של כמה zooms יכול לעזור.
   - המסקנה הייתה שהשילוב המועיל הוא **local multiscale**,
     לא broad scene-scale fusion.

3. Study 04:
   - עברנו משאלה של “איזה feature בודד הכי טוב?”
     לשאלה של “האם מודל depth-only נלמד יכול להוציא סיגנל חזק יותר?”
   - שם השתמשנו ב-`30` תמונות לכל stratum, כלומר benchmark מאוזן.
   - המסקנה הייתה ש-`random_forest_top_24` ניצח את best single feature,
     ושהמשפחה החזקה עדיין מרוכזת סביב:
     - `bbox_only`
     - `bbox_expand_1_5x`
     - `bbox_expand_2x`

Study 06 לוקח את המסקנות האלה ומבצע צעד המשך:

- לא לרוץ על מדגם קטן ומאוזן
- לא להשתמש בכל `108` הפיצ'רים של Study 04
- אלא לבדוק אם אפשר לקבל מודל חזק גם עם:
  - כל הדאטה
  - subset קטן יותר של contexts
  - subset קטן יותר של aggregations
  - score field אחד בלבד

## המוטיבציה

ל-Study 06 היו שלוש מטרות:

1. לבדוק generalization על כל הדאטה

Study 04 עבד על `1,440` תמונות בלבד.
זה מספיק בשביל benchmark טוב, אבל עדיין לא משקף את התפלגות הדאטה
האמיתית במלואה.

2. לבדוק אם צריך באמת feature space גדול

Study 04 השתמש ב:

- `9` contexts
- `3` aggregations
- `4` score fields
- סך הכל `108` features

זה חזק, אבל גם מורכב.
רצינו לבדוק אם raw object depth בלבד כבר מספיק כדי לקבל depth-only model
טוב.

3. לבדוק האם המסקנות הקודמות יציבות

בפרט:

- האם `bbox_only` עדיין מנצח?
- האם `bbox_midpoint` עדיין aggregation חזק?
- האם local zooms עדיין עדיפים על `full_image`?
- האם random forest עדיין נותן gain ברור מעל single feature?

## מה בדיוק השתמשנו כאן

הקונפיגורציה המלאה נמצאת ב-[config.yaml](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/config.yaml).

### כל הדאטה

במקום לבחור `30` תמונות לכל stratum כמו ב-Study 04,
Study 06 משתמש ב:

- **כל** התמונות בדאטה
- `48` strata של `distance x weather x time`
- `15,064` תמונות סך הכל

מתוך [artifacts/features/summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/features/summary.json):

- `num_selected_samples = 15064`
- `num_strata = 48`
- `mean_samples_per_stratum = 313.83`

### contexts

השתמשנו רק ב-contexts הבאים:

- `bbox_only`
- `bbox_expand_1_5x`
- `bbox_expand_2x`
- `bbox_expand_4x`
- `full_image`

זו תת-קבוצה מכוונת של ה-contexts החזקים או המעניינים ביותר מהסטאדיז
הקודמים.

### aggregation methods

השתמשנו בשלוש צורות סיכום depth:

- `bbox_midpoint`
- `bbox_mean`
- `inner50_median`

כלומר:

- `middle`
- `mean`
- `median`

### score fields

כאן עשינו פישוט משמעותי:

- השתמשנו **רק** ב-`object_depth`

לא השתמשנו ב:

- `object_depth_percentile_5_95`
- `object_minus_ring`
- `object_minus_ring_normalized`

המטרה הייתה לבנות ניסוי פשוט, ממוקד, וקל יותר לפרשנות.

### כמה features יש

בסך הכל:

- `5 contexts x 3 aggregations x 1 score field = 15` features

זאת לעומת `108` features ב-Study 04.

## מה הקוד עושה בפועל

קובץ ההרצה הראשי:

- [run_all_images_midpoint_random_forest.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/run_all_images_midpoint_random_forest.py)

הקוד מחולק לשני שלבים עיקריים:

### שלב 1: בניית feature cache לכל הדאטה

השלב הזה:

1. טוען את כל התמונות מה-dataset
2. מגדיר את רשימת ה-contexts וה-aggregations מה-config
3. בודק אילו representation rows כבר קיימים מסטאדיז קודמים
4. מחשב רק את השורות החסרות
5. שומר הכל ל-CSV

בפועל נעשה reuse של rows מ-Study 04:

- `reused_rows = 21600`

וחישוב מחדש של:

- `missing_rows_to_compute = 204360`

מתוך:

- `total_expected_rows = 225960`

### שלב 2: בניית fused feature table ואימון random forest

לאחר שיש representation rows:

1. הקוד הופך את הטבלה הארוכה לטבלה רחבה
2. כל שורה מייצגת תמונה אחת
3. כל feature הוא עמודה
4. נבנית טבלת fused features אחת
5. נבדקים קודם כל single features
6. לאחר מכן מאומן random forest
7. נשמרים metrics, feature importances ו-predictions

### פרוטוקול ההערכה

כמו ב-Study 04:

- `5-fold` cross-validation
- folds מאוזנים לפי `stratum_key`
- ההערכה היא out-of-fold

כלומר:

- כל prediction שמופיע בדוחות התקבל על תמונות שלא היו ב-train fold שלהן

## מה תיקנו כדי שהריצה תהיה יציבה

Study 06 היה כבד בהרבה מ-Study 04, ולכן במהלך הפיתוח הוספנו שני מנגנוני
יציבות חשובים:

### 1. טיפול ב-bounding boxes קטנים

הייתה קריסה על detections קטנים או דקים מאוד, שבהם shrink של bbox סביב
המרכז היה יוצר bbox לא חוקי.

הפתרון שנוסף ב-[representation.py](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/pipeline/depth/representation.py):

- אם `focus_bbox` קורס בזמן resize:
  - נופלים חזרה ל-`full_bbox`
- אם `surrounding_bbox` קורס בזמן expand:
  - גם שם נופלים חזרה ל-`full_bbox`

כך הניסוי לא נכשל בגלל edge cases גיאומטריים.

### 2. שמירת התקדמות תוך כדי ריצה

בתחילת הפיתוח ה-cache נשמר רק בסוף.
זה היה מסוכן כי אם הריצה נעצרת, כל מה שחושב בזיכרון הולך לאיבוד.

הקוד שונה כך ש:

- `representation_records.csv` נכתב ומעודכן תוך כדי ריצה
- כל sample חדש מחושב ומתווסף לדיסק
- `summary.json` מתעדכן כ-checkpoint
- rerun של אותו command יכול להמשיך ממה שכבר נשמר

כלומר:

- אם הריצה נעצרת באמצע, לא צריך להתחיל מאפס

## אילו קבצים נוצרים

### artifacts/features

- [selected_samples.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/features/selected_samples.csv)
  - רשימת כל התמונות ששייכות לסטאדי
  - כאן זה בפועל כל הדאטה

- [representation_records.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/features/representation_records.csv)
  - הטבלה הארוכה
  - כל תמונה מופיעה פעם אחת לכל `(context, aggregation)`

- [summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/features/summary.json)
  - סיכום metadata על ה-cache, גודל הדאטה, reuse, strata וכו'

### artifacts/fused_features

- [depth_only_feature_table.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/fused_features/depth_only_feature_table.csv)
  - טבלה רחבה
  - שורה אחת לכל תמונה
  - `15` פיצ'רים כעמודות

### artifacts/reports

- [single_feature_cv_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/single_feature_cv_metrics.csv)
  - דירוג של כל feature בודד

- [subset_random_forest_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_metrics.csv)
  - טבלת המודלים

- [subset_random_forest_feature_importances.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_feature_importances.csv)
  - חשיבות הפיצ'רים במודל

- [subset_random_forest_predictions.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_predictions.csv)
  - תחזית לכל תמונה, כולל signed ו-absolute errors

- [summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/summary.json)
  - תקציר של התוצאה הסופית

## התוצאות המרכזיות

מתוך [artifacts/reports/summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/summary.json):

- `num_image_rows = 15064`
- `num_available_features = 15`

### הפיצ'ר הבודד הטוב ביותר

- `bbox_only__bbox_midpoint__object_depth`
- `CV MAE = 24.21m`

### המודל הטוב ביותר

- `random_forest_top_24`
- בפועל הוא משתמש בכל `15` הפיצ'רים הזמינים
- `CV MAE = 20.92m`

### מה המשמעות

זה אומר:

- גם על כל הדאטה
- גם אחרי שצמצמנו את feature space מאוד
- עדיין יש gain ברור מלמידה לא-ליניארית depth-only

השיפור מול best single feature:

- `24.21m -> 20.92m`
- שיפור של `3.29m`
- בערך `13.6%` שיפור ב-MAE

## מה אומר דירוג ה-single features

הקובץ:

- [single_feature_cv_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/single_feature_cv_metrics.csv)

ה-top features הם:

1. `bbox_only__bbox_midpoint__object_depth`
2. `bbox_expand_1_5x__bbox_midpoint__object_depth`
3. `bbox_only__inner50_median__object_depth`
4. `bbox_expand_2x__bbox_midpoint__object_depth`

לעומת זאת, החלשים ביותר הם:

- כל ה-`bbox_mean`
- וכל ה-`full_image`

### מסקנות

1. `bbox_midpoint` הוא ה-aggregation החזק ביותר
2. `inner50_median` שני
3. `bbox_mean` חלש משמעותית
4. tight local contexts עדיפים בבירור על `full_image`

אם מסתכלים ברמת context:

- הטוב ביותר: `bbox_only`
- אחריו: `1.5x`
- אחריו: `2x`
- אחריו: `4x`
- האחרון: `full_image`

זה מאוד עקבי עם הסטאדיז הקודמים.

## מה אומר random forest

הקובץ:

- [subset_random_forest_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_metrics.csv)

יש כאן שני מודלים:

- `random_forest_top_24`
- `random_forest_top_12`

בפועל:

- `top_24` משתמש בכל `15` features
- `top_12` משתמש ב-12 הטובים ביותר

התוצאה:

- all `15` features:
  - `MAE = 20.92`
  - `RMSE = 26.83`
  - `R2 = 0.505`

- top `12` features:
  - `MAE = 21.14`
  - `RMSE = 27.12`
  - `R2 = 0.494`

### המשמעות

- שלושת הפיצ'רים הנוספים לא משנים את התמונה כולה,
  אבל כן עוזרים קצת
- כלומר גם פיצ'רים חלשים יחסית יכולים להיות מועילים בתוך ensemble,
  אפילו אם הם לא מנצחים לבד

## מה אומרות ה-feature importances

הקובץ:

- [subset_random_forest_feature_importances.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_feature_importances.csv)

הפיצ'רים הכי חשובים במודל:

1. `bbox_only__bbox_midpoint__object_depth`
2. `bbox_expand_1_5x__bbox_midpoint__object_depth`
3. `bbox_only__inner50_median__object_depth`
4. `bbox_expand_2x__bbox_midpoint__object_depth`

אם מחברים חשיבות לפי context:

- `bbox_only` מוביל בפער
- אחריו `1.5x`
- אחריו `2x`
- אחריו `4x`
- אחרון `full_image`

אם מחברים לפי aggregation:

- `bbox_midpoint` הוא החזק ביותר
- `inner50_median` שני
- `bbox_mean` שלישי

### המשמעות

המודל לא “המציא” אסטרטגיה חדשה.
הוא חיזק את מה שכבר למדנו:

- local contexts הם הליבה
- midpoint הוא summary מצוין
- `full_image` תורם מעט

זה סימן טוב של consistency מחקרית.

## מה אומרות התחזיות ברמת המרחק

הקובץ:

- [subset_random_forest_predictions.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_predictions.csv)

כשמסתכלים לפי true distance, מתקבל דפוס מאוד ברור:

- ב-`20m` המודל עושה overprediction חזק
- ב-`30m` עד `70m` הוא עדיין נוטה להעריך מרחק גדול מדי
- באזור `80m` עד `100m` הוא הכי יציב
- מ-`100m` ומעלה הוא מתחיל לעשות underprediction
- ב-`150m` השגיאה כבר גדולה מאוד

דוגמאות:

- `20m`: `MAE ≈ 25.10m`, bias חיובי חזק
- `80m`: `MAE ≈ 14.34m`
- `90m`: `MAE ≈ 13.36m`
- `125m`: `MAE ≈ 26.12m`, bias שלילי חזק
- `150m`: `MAE ≈ 45.82m`, underprediction קיצוני

### המשמעות

זה אומר שהמודל:

- מצליח ללמוד depth signal אמיתי
- אבל עדיין סובל מ-distance compression

במילים אחרות:

- relative depth alone לא הופך ל-metric depth מושלם
- גם random forest טוב עדיין מתקשה מאוד בקצוות

## מה אומרות התחזיות לפי תנאי צילום

לפי weather:

- `clear_sky`: `MAE ≈ 20.54m`
- `light_rain`: `MAE ≈ 21.30m`

לפי time:

- `10AM`: `MAE ≈ 20.12m`
- `8PM`: `MAE ≈ 22.50m`

השילוב הקשה ביותר:

- `light_rain + 8PM`: `MAE ≈ 24.04m`

### המשמעות

- time of day כנראה משפיע יותר מהמזג עצמו
- `8PM` קשה משמעותית יותר מ-`10AM`
- `light_rain + 8PM` הוא corner case מאתגר במיוחד

## השוואה ישירה ל-Study 04

קובץ ההשוואה המרכזי:

- [Study 04 README](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_04_depth_only_models_30_per_stratum/README.md)
- [Study 04 summary](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_04_depth_only_models_30_per_stratum/artifacts/reports/depth_only_study_summary.json)

### Study 04

- `1440` תמונות
- `108` features
- best single:
  - `25.04m`
- best RF:
  - `22.95m`
- improvement vs single:
  - `2.09m`

### Study 06

- `15064` תמונות
- `15` features
- best single:
  - `24.21m`
- best RF:
  - `20.92m`
- improvement vs single:
  - `3.29m`

### מה זה אומר

1. אותו single feature עדיין מנצח

- `bbox_only__bbox_midpoint__object_depth`

כלומר המסקנה הכי חשובה מ-Study 04 לא נשברה.

2. גם feature space פשוט יותר עדיין עובד

הוצאנו:

- percentile features
- ring features
- normalized variants

ועדיין קיבלנו מודל חזק.

3. המספרים של Study 06 טובים יותר

זה מעודד, אבל צריך לפרש בזהירות:

- Study 04 הוא balanced benchmark
- Study 06 הוא full-dataset benchmark

לכן זו לא השוואת apples-to-apples מלאה.

ועדיין:

- עצם זה שהביצועים נשארו טובים, ואפילו השתפרו,
  מחזק את המסקנה שה-depth-only signal הוא אמיתי ויציב.

## מה המסקנה המחקרית

Study 06 תומך בכמה מסקנות חשובות:

1. raw `object_depth` alone כבר מכיל depth signal חזק
2. local context חשוב יותר מ-global context
3. `bbox_midpoint` נשאר aggregator מצוין
4. random forest מצליח לנצל שילוב של כמה local depth views טוב יותר מכל single feature
5. המודל עדיין מתקשה בקצוות המרחק, במיוחד קרוב מאוד ורחוק מאוד

כלומר:

- Study 04 הראה שאפשר ללמוד lower model depth-only
- Study 06 הראה שזה לא artifact של sample קטן או feature engineering כבד

## מגבלות הסטאדי

למרות התוצאה הטובה, יש מגבלות ברורות:

1. הדאטה ב-Study 06 אינו מאוזן בין strata
2. עדיין לא השתמשנו ב-geometry features
3. relative depth alone עדיין compresses distances
4. אין כאן עדיין analysis package ויזואלי כמו ב-Study 04

## איך מריצים

בדיקת setup בלבד:

```powershell
.\.venv\Scripts\python.exe "attempts/second attempt/studies/study_06_all_images_midpoint_random_forest/run_all_images_midpoint_random_forest.py" --dry-run
```

הרצה מלאה:

```powershell
.\.venv\Scripts\python.exe "attempts/second attempt/studies/study_06_all_images_midpoint_random_forest/run_all_images_midpoint_random_forest.py"
```

## מה כדאי לבדוק מיד אחרי ריצה

אם רוצים לבדוק מהר את השורה התחתונה:

1. [artifacts/reports/summary.json](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/summary.json)
2. [artifacts/reports/subset_random_forest_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_metrics.csv)
3. [artifacts/reports/single_feature_cv_metrics.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/single_feature_cv_metrics.csv)

אם רוצים להבין את עומק התוצאה:

4. [artifacts/reports/subset_random_forest_feature_importances.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_feature_importances.csv)
5. [artifacts/reports/subset_random_forest_predictions.csv](/C:/Users/depthlev/Desktop/Z_Drone_Final_Project_GA/attempts/second%20attempt/studies/study_06_all_images_midpoint_random_forest/artifacts/reports/subset_random_forest_predictions.csv)

## השלב הטבעי הבא

המשך ישיר ל-Study 06 יכול להיות אחד מהבאים:

- לבנות `analyze_study_06.py` עם גרפים ו-analysis package
- להוסיף calibration-by-distance analysis
- להשוות depth-only lower model עם מודל שמקבל גם geometry
- לבדוק stacked model שבו Study 06 משמש lower signal ולא prediction סופי
