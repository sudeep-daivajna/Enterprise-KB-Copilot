"use client";

import { useState } from "react";
import type { SourceItem } from "../types";

export default function Sources({ sources }: { sources: SourceItem[] }) {
  const [open, setOpen] = useState(false);

  if (!sources?.length) return null;

  return (
    <div className="mt-3 border border-gray-800 rounded-xl bg-gray-950">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm text-gray-300"
      >
        <span>Sources ({sources.length})</span>
        <span className="text-gray-500">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-3">
          {sources.map((s, idx) => (
            <div
              key={s.chunk_id}
              className="p-3 rounded-lg border border-gray-800 bg-gray-950"
            >
              <div className="text-xs text-gray-400">
                [{idx + 1}] {s.title}
              </div>

              <div className="text-xs text-gray-500 break-all">{s.source}</div>

              <div className="mt-2 text-sm text-gray-200">{s.snippet}</div>

              <div className="mt-1 text-[11px] text-gray-600">
                chunk_id: {s.chunk_id}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
