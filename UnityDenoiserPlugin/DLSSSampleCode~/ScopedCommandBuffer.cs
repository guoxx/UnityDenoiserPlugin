using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityDenoiserPlugin
{
    public ref struct ScopedCommandBuffer
    {
        private static CommandBuffer s_commands = new CommandBuffer()
        {
            name = ""
        };

        private string m_scopeName;

        public ScopedCommandBuffer(string name)
        {
            m_scopeName = name;
            s_commands.BeginSample(m_scopeName);
        }

        public CommandBuffer Get()
        {
            return s_commands;
        }

        public static implicit operator CommandBuffer(ScopedCommandBuffer c)
        {
            return s_commands;
        }

        public void Dispose()
        {
            s_commands.EndSample(m_scopeName);
            m_scopeName = null;

            Graphics.ExecuteCommandBuffer(s_commands);
            s_commands.Clear();
        }
    }
}
