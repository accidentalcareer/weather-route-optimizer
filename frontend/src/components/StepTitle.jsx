/** Numbered workflow step heading: 1 Forecast, 2 Origin & destination, ... */
export default function StepTitle({ n, id, children, optional }) {
  return (
    <h2 className="step__title" id={id}>
      <span className="step__num" aria-hidden="true">
        {n}
      </span>
      <span>{children}</span>
      {optional && <span className="step__optional">optional</span>}
    </h2>
  );
}
