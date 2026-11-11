using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>One update from run.py. Field names must match the JSON keys.</summary>
[Serializable]
public class EmgPacket
{
    public int seq;
    public float time;
    public float latency_ms;
    public float[] fused;         // normalised 0..1
    public float[] abs;
    public float[] delta;
    public float[] truth;         // empty when the source has no glove data
    public float[] fused_angles;  // glove units, close to degrees
    public float[] abs_angles;
    public float[] truth_angles;
}

/// <summary>Listens for packets on a background thread and exposes the newest one on the main thread.</summary>
public class EmgUdpReceiver : MonoBehaviour
{
    public int port = 5005;

    public EmgPacket Latest { get; private set; }
    public float SecondsSinceLast { get; private set; } = float.PositiveInfinity;

    private UdpClient client;
    private Thread thread;
    private volatile bool running;
    private readonly object gate = new object();
    private string pending;

    private void OnEnable()
    {
        client = new UdpClient(port);
        running = true;
        thread = new Thread(Listen) { IsBackground = true, Name = "EmgUdpReceiver" };
        thread.Start();
    }

    private void Listen()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                string text = Encoding.UTF8.GetString(client.Receive(ref remote));
                lock (gate) pending = text;
            }
            catch (SocketException)
            {
                // Raised when the socket closes, and on Windows after an ICMP port-unreachable.
            }
            catch (ObjectDisposedException)
            {
                return;
            }
        }
    }

    private void Update()
    {
        string text;
        lock (gate)
        {
            text = pending;
            pending = null;
        }

        SecondsSinceLast += Time.unscaledDeltaTime;
        if (text == null) return;

        try
        {
            Latest = JsonUtility.FromJson<EmgPacket>(text);
            SecondsSinceLast = 0f;
        }
        catch (ArgumentException e)
        {
            Debug.LogWarning("Ignoring a malformed EMG packet: " + e.Message);
        }
    }

    private void OnDisable()
    {
        running = false;
        client?.Close();
        thread?.Join(500);
        client = null;
        thread = null;
    }
}
