# 🔥 Finalized Suspicious Behavior Set (CPU Safe)

We’re choosing behaviors that are:
- visually clear
- computationally light
- rule-based
- demo-safe

## ✅ 1️⃣ Loitering Detection

If:
- Person inside frame
- Movement < threshold
- Time > 8 seconds

Then → suspicious

**Implementation:**
Track center displacement over time.

Cheap. Reliable. Looks smart.

## ✅ 2️⃣ Abandoned Bag Detection

**Detect:**
- backpack
- handbag
- suitcase

**Logic:**
- Bag appears near person
- Person moves > X pixels away
- Bag stays stationary for 5 sec

Then → abandoned object alert.

THIS will impress judges.

## ✅ 3️⃣ Phone Near Face Detection

**Detect:**
- person
- cell phone

If phone box overlaps upper half of person box for > 5 sec → suspicious distraction

No MediaPipe required.

CPU safe.

---

## 🧱 System Architecture (Final)

```
YOLOv8n (restricted classes)
    ↓
Ultralytics built-in tracker
    ↓
ID dictionary store
    ↓
Behavior rules engine
    ↓
Suspicion score
    ↓
Overlay
```

---

## 🧮 Suspicion Scoring System

Each person ID has:

```python
person_data = {
    "first_seen": time,
    "last_moved": time,
    "loitering_flag": False,
    "phone_flag": False,
    "score": 0
}
```

**Scoring:**
- +2 = loitering
- +3 = abandoned object
- +1 = phone misuse

If score ≥ 3 → RED BOX

---

## 🧠 Optimization Tricks For Your CPU

Do these or suffer:

**Use:**
```python
model = YOLO("yolov8n.pt")
model.fuse()
```

**Set:**
```python
imgsz=640
conf=0.4
iou=0.5
classes=[0,24,26,28,67]  # person, backpack, handbag, suitcase, cell phone
```

**Use:**
```python
stream=True
```

Disable fancy drawing.

Avoid Python-heavy nested loops.
Use dictionaries indexed by ID.

---

## 🎬 Exhibition Flow Script

You rehearse this.

**Step 1:**
Stand normally → green box

**Step 2:**
Stand still for 10 sec → yellow warning

**Step 3:**
Drop bag → walk 2 meters away → RED ALERT

**Step 4:**
Hold phone to face → suspicious tag

Boom.

You look like you built mini airport security AI.

---

## 🚨 Important: Thermal Reality

Your CPU is 54°C idle already.

During demo:
- Plug charger
- Disable power saver
- Close browser tabs
- Use a cooling pad if possible

U-series CPUs throttle hard.

---

## ⏱ Build Plan Starting Now

**Hour 1–2:**
Basic detection + tracking + FPS counter

**Hour 3–4:**
Implement loitering logic

**Hour 5–6:**
Implement abandoned bag logic

**Hour 7:**
Phone overlap logic

**Hour 8:**
UI polish + rehearsal
