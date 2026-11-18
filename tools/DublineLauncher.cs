using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

[assembly: AssemblyTitle("Dubline")]
[assembly: AssemblyProduct("Dubline")]
[assembly: AssemblyDescription("Windows launcher for Dubline")]

internal static class DublineLauncher
{
    private static int Main(string[] args)
    {
        var start = new ProcessStartInfo
        {
            FileName = Path.Combine(Environment.SystemDirectory, "cmd.exe"),
            Arguments = "/d /c start.bat",
            WorkingDirectory = AppDomain.CurrentDomain.BaseDirectory,
            UseShellExecute = false
        };

        if (args.Length > 0)
        {
            var encoded = Array.ConvertAll(args, arg =>
                Convert.ToBase64String(Encoding.UTF8.GetBytes(arg)));
            start.EnvironmentVariables["DUBLINE_EXE_ARGS"] = ":" + string.Join(",", encoded);
        }

        Console.CancelKeyPress += (sender, e) => e.Cancel = true;
        try
        {
            using (var process = Process.Start(start))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            Console.Error.WriteLine("Dubline could not start: " + e.Message);
            return 1;
        }
    }
}
