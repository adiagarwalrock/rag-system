import { ZodError } from "zod";

export class ApiError extends Error {
  readonly status?: number;
  readonly path: string;
  readonly body?: unknown;
  readonly validation?: ZodError;

  constructor({
    message,
    status,
    path,
    body,
    validation,
  }: {
    message: string;
    status?: number;
    path: string;
    body?: unknown;
    validation?: ZodError;
  }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.body = body;
    this.validation = validation;
  }
}

export function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Unknown error";
}
