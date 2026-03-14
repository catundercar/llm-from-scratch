export function generateStaticParams() {
  // Phase -> lesson count mapping
  const phaseLessons: Record<number, number[]> = {
    0: [1, 2, 3],
    1: [1, 2],
    2: [1],
    3: [1, 2],
    4: [1],
    5: [1],
    6: [1],
    7: [1],
    8: [1],
    9: [1],
  };

  const params: { lessonId: string }[] = [];
  for (const [, lessons] of Object.entries(phaseLessons)) {
    for (const lessonId of lessons) {
      params.push({ lessonId: String(lessonId) });
    }
  }
  return params;
}

export default function LessonLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
