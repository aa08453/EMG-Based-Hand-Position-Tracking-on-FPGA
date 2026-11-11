using UnityEngine;

/// <summary>
/// Builds a simple hand from capsules and drives it from EmgUdpReceiver.
/// Add this to an empty GameObject at the origin and press Play. No model or rig needed.
/// </summary>
[RequireComponent(typeof(EmgUdpReceiver))]
public class EmgProceduralHand : MonoBehaviour
{
    public enum Signal { Fused, AbsoluteOnly, GroundTruth }

    public Signal signal = Signal.Fused;
    public bool rightHand = true;

    [Tooltip("Higher follows the 20 Hz updates more tightly, lower looks smoother.")]
    public float smoothing = 15f;

    [Tooltip("Multiplies every joint angle.")]
    public float angleGain = 1f;
    public float minAngle = -15f;
    public float maxAngle = 110f;

    [Tooltip("Packet index for: thumb MCP, thumb IP, index MCP, index PIP, middle MCP, middle PIP, ring MCP, ring PIP, little MCP, little PIP.")]
    public int[] jointOrder = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    private const int JointCount = 10;
    private readonly Transform[] proximal = new Transform[5];
    private readonly Transform[] middle = new Transform[5];
    private readonly Transform[] distal = new Transform[5];
    private readonly Quaternion[] restRotation = new Quaternion[5];
    private readonly float[] current = new float[JointCount];
    private EmgUdpReceiver receiver;
    private Vector3 thumbAxis;

    private void Awake()
    {
        receiver = GetComponent<EmgUdpReceiver>();
        float side = rightHand ? 1f : -1f;
        thumbAxis = new Vector3(-0.4f, 0f, side).normalized;

        // Palm faces -Z, towards the default camera. Fingers point along +Y.
        Segment(transform, new Vector3(0f, 0f, 0f), new Vector3(2.2f, 2.4f, 0.5f), PrimitiveType.Cube);

        float[] x = { 0.8f, 0.27f, -0.27f, -0.8f };
        float[] length = { 0.8f, 0.9f, 0.82f, 0.62f };
        for (int f = 0; f < 4; f++)
        {
            Transform root = Pivot("Finger" + (f + 1), transform, new Vector3(x[f] * side, 1.2f, 0f), Quaternion.identity);
            BuildDigit(f + 1, root, length[f], 0.42f);
        }

        Transform thumb = Pivot("Thumb", transform, new Vector3(1.05f * side, -0.5f, -0.1f), Quaternion.Euler(0f, 0f, -50f * side));
        BuildDigit(0, thumb, 0.75f, 0.48f);
    }

    private void BuildDigit(int digit, Transform root, float len, float width)
    {
        proximal[digit] = root;
        Capsule(root, len, width);
        middle[digit] = Pivot("Middle", root, new Vector3(0f, len, 0f), Quaternion.identity);
        float len2 = len * 0.75f;
        Capsule(middle[digit], len2, width * 0.92f);
        distal[digit] = Pivot("Distal", middle[digit], new Vector3(0f, len2, 0f), Quaternion.identity);
        Capsule(distal[digit], len * 0.6f, width * 0.85f);
        restRotation[digit] = root.localRotation;
    }

    private static Transform Pivot(string name, Transform parent, Vector3 position, Quaternion rotation)
    {
        var pivot = new GameObject(name).transform;
        pivot.SetParent(parent, false);
        pivot.localPosition = position;
        pivot.localRotation = rotation;
        return pivot;
    }

    private static void Capsule(Transform parent, float length, float width)
    {
        Segment(parent, new Vector3(0f, length * 0.5f, 0f), new Vector3(width, length * 0.5f, width), PrimitiveType.Capsule);
    }

    private static void Segment(Transform parent, Vector3 position, Vector3 scale, PrimitiveType type)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        Destroy(go.GetComponent<Collider>());
        go.transform.SetParent(parent, false);
        go.transform.localPosition = position;
        go.transform.localScale = scale;
        go.GetComponent<Renderer>().material.color = new Color(0.87f, 0.68f, 0.56f);
    }

    private float[] SelectAngles(EmgPacket packet)
    {
        switch (signal)
        {
            case Signal.AbsoluteOnly: return packet.abs_angles;
            case Signal.GroundTruth: return packet.truth_angles;
            default: return packet.fused_angles;
        }
    }

    private void Update()
    {
        EmgPacket packet = receiver.Latest;
        if (packet == null) return;
        float[] angles = SelectAngles(packet);
        if (angles == null || angles.Length < JointCount) return;

        float t = 1f - Mathf.Exp(-smoothing * Time.deltaTime);
        for (int j = 0; j < JointCount; j++)
        {
            float target = Mathf.Clamp(angles[jointOrder[j]] * angleGain, minAngle, maxAngle);
            current[j] = Mathf.Lerp(current[j], target, t);
        }

        for (int digit = 0; digit < 5; digit++)
        {
            float first = current[digit * 2];
            float second = current[digit * 2 + 1];
            Vector3 axis = digit == 0 ? thumbAxis : Vector3.left;
            proximal[digit].localRotation = restRotation[digit] * Quaternion.AngleAxis(first, axis);
            middle[digit].localRotation = Quaternion.AngleAxis(second, axis);
            // The model has no outermost joint, so it follows the one before it.
            distal[digit].localRotation = Quaternion.AngleAxis(second * 0.66f, axis);
        }
    }

    private void OnGUI()
    {
        string status = receiver.SecondsSinceLast > 1f
            ? "Waiting for data. Is run.py running?"
            : "Frame " + receiver.Latest.seq + "   inference " + receiver.Latest.latency_ms.ToString("F1") + " ms";
        if (signal == Signal.GroundTruth && receiver.Latest != null
            && (receiver.Latest.truth_angles == null || receiver.Latest.truth_angles.Length == 0))
        {
            status += "\nThis source has no ground truth.";
        }
        GUI.Label(new Rect(10, 10, 500, 50), status);
    }
}
