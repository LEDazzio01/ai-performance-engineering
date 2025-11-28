'use client';

import useSWR, { type SWRConfiguration } from 'swr';
import useSWRMutation, { type SWRMutationConfiguration } from 'swr/mutation';
import { APIError } from './api';

export function getErrorMessage(error: unknown, fallback = 'Something went wrong'): string {
  if (!error) return fallback;
  if (error instanceof APIError) return error.message;
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return fallback;
}

export function useApiQuery<T>(
  key: string | readonly unknown[] | null,
  fetcher: () => Promise<T>,
  options?: SWRConfiguration<T, Error>
) {
  return useSWR<T, Error>(key, key ? () => fetcher() : null, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
    ...options,
  });
}

export function useApiMutation<TPayload, TResult>(
  key: string,
  fetcher: (payload: TPayload) => Promise<TResult>,
  options?: SWRMutationConfiguration<TResult, Error, string, TPayload>
) {
  return useSWRMutation<TResult, Error, string, TPayload>(key, (_, { arg }) => fetcher(arg), {
    throwOnError: true,
    ...options,
  });
}
